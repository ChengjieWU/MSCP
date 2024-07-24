from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
import numpy as np
import gym

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_schedule_fn, update_learning_rate, polyak_update


class GuidedSACPolicy(SACPolicy):
    def make_critic_guided(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self.critic_kwargs.copy()
        critic_kwargs['n_critics'] = 1
        critic_kwargs = self._update_features_extractor(critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)

        if self.share_features_extractor:
            self.critic_guided = self.make_critic_guided(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic_guided.named_parameters() if
                                 "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic_guided = self.make_critic_guided(features_extractor=None)
            critic_parameters = self.critic_guided.parameters()

        self.critic_guided.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)


class GuidedSAC:
    def __init__(self,
                 policy: GuidedSACPolicy,
                 device,
                 learning_rate: Union[float, Schedule] = 3e-4,
                 ent_coef: Union[str, float] = "auto",
                 target_entropy: Union[str, float] = "auto",
                 action_space: Optional[gym.spaces.Space] = None,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 coe_env_r: float = 1.,
                 coe_int_r: float = 1.,
                 ablation_sac_reward_sum: bool = False,
                 ) -> None:
        self.policy = policy
        self.device = device
        self.learning_rate = learning_rate

        self.target_entropy = target_entropy
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.lr_schedule = get_schedule_fn(self.learning_rate)

        self._create_aliases()
        self._setup_entropy(action_space)

        self.gamma = gamma
        self.tau = tau
        self.coe_env_r = coe_env_r
        self.coe_int_r = coe_int_r
        self.ablation_sac_reward_sum = ablation_sac_reward_sum

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.critic_guided = self.policy.critic_guided
    
    def _setup_entropy(self, action_space: Optional[gym.spaces.Space] = None) -> None:
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = torch.tensor(float(self.ent_coef)).to(self.device)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the model's action(s) from an observation

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        return self.policy.predict(observation, state, mask, deterministic)
    
    def set_training_mode(self, mode: bool) -> None:
        return self.policy.set_training_mode(mode)

    def update(self, batch):
        # Switch to train mode (this affects batch norm / dropout)
        self.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        # optimizers.append(self.critic_guided.optimizer)   # guided SAC does NOT add this
        self._update_learning_rate(optimizers)

        # ------------------------------------------
        # Prepare input
        # subgoals = batch['traj_subgoals']
        # states = batch['observations'].copy()
        # states[:, :2] = states[:, :2] - subgoals[:, :2]
        # next_states = batch['next_observations'].copy()
        # next_states[:, :2] = next_states[:, :2] - subgoals[:, :2]

        states = torch.from_numpy(batch['observations']).to(self.device)
        next_states = torch.from_numpy(batch['next_observations']).to(self.device)
        actions = torch.from_numpy(batch['actions']).to(self.device)
        rewards_env = torch.from_numpy(batch['rewards']).to(self.device).unsqueeze(dim=1)
        # rewards_in = torch.from_numpy(batch['intrinsic_rewards']).to(self.device).unsqueeze(dim=1)
        rewards_in = torch.from_numpy(batch['guided_rewards']).to(self.device).unsqueeze(dim=1)
        masks = torch.from_numpy(batch['masks']).to(self.device).unsqueeze(dim=1)

        if self.ablation_sac_reward_sum:
            rewards_env = self.coe_env_r * rewards_env + self.coe_int_r * rewards_in
        
        # ------------------------------------------
        # Action by the current actor for the sampled state
        actions_pi, log_prob = self.actor.action_log_prob(states)
        log_prob = log_prob.reshape(-1, 1)

        # ------------------------------------------
        # Compute entropy coefficient loss
        ent_coef_loss = None
        if self.ent_coef_optimizer is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = torch.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
        else:
            ent_coef = self.ent_coef_tensor

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None:
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

        # ------------------------------------------
        # Compute critic_loss loss
        with torch.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(next_states)
            # Compute the next Q values: min over all critics targets
            next_q_values = torch.cat(self.critic_target(next_states, next_actions), dim=1)
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            target_q_values = rewards_env + masks * self.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = self.critic(states, actions)

        # Compute critic loss
        critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # ------------------------------------------
        # Compute guiding critic loss
        if not self.ablation_sac_reward_sum and self.coe_int_r > 0:
            # SAC with guided reward: self.coe_int_r > 0 and self.ablation_sac_reward_sum = True
            # Pure SAC: self.coe_int_r = 0 and self.ablation_sac_reward_sum = False

            # Compute critic_guided loss
            current_q_in = self.critic_guided(states, actions)[0]
            critic_guided_loss = F.mse_loss(current_q_in, rewards_in)

            # Optimize the critic_guided
            self.critic_guided.optimizer.zero_grad()
            critic_guided_loss.backward()
            self.critic_guided.optimizer.step()
        else:
            critic_guided_loss = torch.tensor(0.0).to(self.device)

        # ------------------------------------------
        # Compute actor loss
        # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
        # Min over all critic networks
        q_values_pi = torch.cat(self.critic.forward(states, actions_pi), dim=1)
        min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)

        if self.ablation_sac_reward_sum:
            q_pi = min_qf_pi
        else:
            q_in_pi = self.critic_guided.forward(states, actions_pi)[0]
            q_pi = self.coe_env_r * min_qf_pi + self.coe_int_r * q_in_pi

        actor_loss = (ent_coef * log_prob - q_pi).mean()

        # Optimize the actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        # info
        info = {
            "ent_coef": ent_coef.item(),
            "q/loss": critic_loss.item(),
            "q/target_q": torch.mean(target_q_values).item(),
            "q/current_q": torch.mean(current_q_values[0]).item(),
            "actor/loss": actor_loss.item(),
            "actor/overall_q": torch.mean(q_pi).item(),
            "actor/env_q": torch.mean(min_qf_pi).item(),
            "actor/log_prob": torch.mean(log_prob).item(),
        }
        if self.ent_coef_optimizer is not None:
            info["ent_coef_loss"] = ent_coef_loss.item()
        if not self.ablation_sac_reward_sum and self.coe_int_r > 0:
            info.update({
                "guide_q/loss": critic_guided_loss.item(),
                "guide_q/current_q": torch.mean(current_q_in).item(),
                "actor/guide_q": torch.mean(q_in_pi).item(),
            })

        return info

    def _update_learning_rate(self, optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        # Log the current learning rate
        # self.logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))
    
    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)


def create_learner(
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        learning_rate: float,
        gamma: float,
        device,
):
    lr_schedule = get_schedule_fn(learning_rate)
    guided_sac_policy = GuidedSACPolicy(observation_space, action_space, lr_schedule=lr_schedule).to(device)
    guided_sac = GuidedSAC(
        guided_sac_policy,
        device,
        learning_rate,
        ent_coef='auto',
        target_entropy='auto',
        action_space=action_space,
        gamma=gamma,
        tau=0.005,
        coe_env_r=1.,
        coe_int_r=3.,
        ablation_sac_reward_sum=False,
    )
    return guided_sac
