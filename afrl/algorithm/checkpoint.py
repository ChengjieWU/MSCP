from pathlib import Path
import pickle
import tempfile

import flax
import wandb
from loguru import logger

from afrl.utils import compare_frozen_dicts


def check_hiql_loading(agent, pretrained_agent, finetune_alg: str, use_rep: bool, visual: bool, use_waypoints: bool, policy_share_value_state: bool):
    assert compare_frozen_dicts(agent.network.params['networks_target_value'], pretrained_agent.network.params['networks_target_value'])
    assert compare_frozen_dicts(agent.network.params['networks_value'], pretrained_agent.network.params['networks_value'])
    assert not compare_frozen_dicts(agent.network.params['networks_actor'], pretrained_agent.network.params['networks_actor'])
    if finetune_alg in {'hiql', 'scpiql'}:
        assert compare_frozen_dicts(agent.network.params['networks_high_actor'], pretrained_agent.network.params['networks_high_actor'])
        if use_rep:
            assert compare_frozen_dicts(agent.network.params['encoders_value_goal'], pretrained_agent.network.params['encoders_value_goal'])
        if visual:
            assert compare_frozen_dicts(agent.network.params['encoders_value_state'], pretrained_agent.network.params['encoders_value_state'])
            if not policy_share_value_state:
                assert not compare_frozen_dicts(agent.network.params['encoders_policy_state'], pretrained_agent.network.params['encoders_policy_state'])
            if not use_waypoints:
                assert not compare_frozen_dicts(agent.network.params['encoders_policy_goal'], pretrained_agent.network.params['encoders_policy_goal'])
            assert compare_frozen_dicts(agent.network.params['encoders_high_policy_state'], pretrained_agent.network.params['encoders_high_policy_state'])
            assert compare_frozen_dicts(agent.network.params['encoders_high_policy_goal'], pretrained_agent.network.params['encoders_high_policy_goal'])
        if finetune_alg == 'scpiql':
            assert compare_frozen_dicts(agent.network.params['networks_fast_high_actor'], pretrained_agent.network.params['networks_fast_high_actor'])
    else:
        if visual:
            assert compare_frozen_dicts(agent.network.params['encoders_value_state'], pretrained_agent.network.params['encoders_value_state'])
            raise RuntimeError("Should not load pretrained policy encoder?")
            assert compare_frozen_dicts(agent.network.params['encoders_policy_state'], pretrained_agent.network.params['encoders_policy_state'])


def load_hiql_checkpoint(agent, pretrained_agent, prefix: str, checkpoint_file: str, wandb_checkpoint_run: str,
                         finetune_alg: str, use_rep: bool, visual: bool, use_waypoints: bool, policy_share_value_state: bool):
    assert (checkpoint_file is None) != (wandb_checkpoint_run is None), "Must specify exactly one of checkpoint_file and wandb_checkpoint_run."
    if checkpoint_file is not None:
        with open(checkpoint_file, 'rb') as fp:
            state_dict = pickle.load(fp)
    else:
        wandb_api = wandb.Api()
        with tempfile.TemporaryDirectory() as tempdirname:
            wandb_api.run(f"{prefix}/{wandb_checkpoint_run}").file(f"checkpoints/params_final.pkl").download(root=tempdirname, replace=False)
            with open(Path(tempdirname).joinpath(f"checkpoints/params_final.pkl"), 'rb') as fp:
                state_dict = pickle.load(fp)

    pretrained_agent = flax.serialization.from_state_dict(pretrained_agent, state_dict['agent'])
    # only load the value funciton and target value function
    params = flax.core.unfreeze(agent.network.params)
    params['networks_target_value'] = pretrained_agent.network.params['networks_target_value']
    params['networks_value'] = pretrained_agent.network.params['networks_value']
    # params['networks_actor'] = pretrained_agent.network.params['networks_actor']    # MUST NOT load actor for training
    if finetune_alg in {'hiql', 'scpiql'}:
        params['networks_high_actor'] = pretrained_agent.network.params['networks_high_actor']
        if use_rep:
            params['encoders_value_goal'] = pretrained_agent.network.params['encoders_value_goal']
        if visual:
            params['encoders_value_state'] = pretrained_agent.network.params['encoders_value_state']
            # params['encoders_policy_state'] = pretrained_agent.network.params['encoders_policy_state']    # MUST NOT load actor
            # params['encoders_policy_goal'] = pretrained_agent.network.params['encoders_policy_goal']
            params['encoders_high_policy_state'] = pretrained_agent.network.params['encoders_high_policy_state']
            params['encoders_high_policy_goal'] = pretrained_agent.network.params['encoders_high_policy_goal']
        if finetune_alg == 'scpiql':
            params['networks_fast_high_actor'] = pretrained_agent.network.params['networks_fast_high_actor']
    else:
        if visual:
            params['encoders_value_state'] = pretrained_agent.network.params['encoders_value_state']
            raise RuntimeError("Should not load pretrained policy encoder?")
            params['encoders_policy_state'] = pretrained_agent.network.params['encoders_policy_state']
    new_network = agent.network.replace(params=flax.core.freeze(params))
    agent = agent.replace(network=new_network)
    check_hiql_loading(agent, pretrained_agent, finetune_alg, use_rep, visual, use_waypoints, policy_share_value_state)
    return agent, pretrained_agent


def load_full_hiql_checkpoint(agent, pretrained_agent, checkpoint_file: str,
                              finetune_alg: str, use_rep: bool, visual: bool, use_waypoints: bool):
    with open(checkpoint_file, 'rb') as fp:
        state_dict = pickle.load(fp)
    pretrained_agent = flax.serialization.from_state_dict(pretrained_agent, state_dict['agent'])
    # load everything
    params = flax.core.unfreeze(agent.network.params)
    params['networks_target_value'] = pretrained_agent.network.params['networks_target_value']
    params['networks_value'] = pretrained_agent.network.params['networks_value']
    params['networks_actor'] = pretrained_agent.network.params['networks_actor']    # ALSO load actor
    if finetune_alg in {'hiql', 'scpiql'}:
        params['networks_high_actor'] = pretrained_agent.network.params['networks_high_actor']
        if use_rep:
            params['encoders_value_goal'] = pretrained_agent.network.params['encoders_value_goal']
        if visual:
            params['encoders_value_state'] = pretrained_agent.network.params['encoders_value_state']
            params['encoders_policy_state'] = pretrained_agent.network.params['encoders_policy_state']    # ALSO load actor
            params['encoders_policy_goal'] = pretrained_agent.network.params['encoders_policy_goal']
            params['encoders_high_policy_state'] = pretrained_agent.network.params['encoders_high_policy_state']
            params['encoders_high_policy_goal'] = pretrained_agent.network.params['encoders_high_policy_goal']
        if finetune_alg == 'scpiql':
            params['networks_fast_high_actor'] = pretrained_agent.network.params['networks_fast_high_actor']
    else:
        if visual:
            params['encoders_value_state'] = pretrained_agent.network.params['encoders_value_state']
            params['encoders_policy_state'] = pretrained_agent.network.params['encoders_policy_state']
    new_network = agent.network.replace(params=flax.core.freeze(params))
    agent = agent.replace(network=new_network)
    # check_hiql_loading(agent, pretrained_agent, finetune_alg, use_rep, visual, use_waypoints)
    return agent, pretrained_agent


def find_pretrained_checkpoint(base_path, conf=None, **kwargs):
    wandb_entity = conf.wandb_entity if conf is not None else kwargs['wandb_entity']
    wandb_project = conf.wandb_project if conf is not None else kwargs['wandb_project']

    env_name = conf.env_name if conf is not None else kwargs['env_name']
    use_rep = conf.use_rep if conf is not None else kwargs['use_rep']
    rep_type = conf.rep_type if conf is not None else kwargs['rep_type']
    seed = conf.seed - 1 if conf is not None else kwargs['seed'] - 1
    way_steps = conf.way_steps if conf is not None else kwargs['way_steps']
    use_icvf = kwargs.get('use_icvf', False)

    run_id = None
    """After pretraining, set the checkpoint finding procedure here.
    
    You can find the run ID by going to the Wandb project page, clicking on the run, and copying the ID from the URL.
    Example:

    if not use_rep:
        if env_name == 'antmaze-umaze-v2':
            run_id = ['9qo01ttb', '4gmezus4', 'z0cyroky'][seed]
        elif env_name == 'antmaze-umaze-diverse-v2':
            run_id = ['tseb9q9b', '0w4moaoo', 'x2ts4mxw'][seed]
    
    After proper setting, the program should be able to find the pretrained checkpoint.
    """

    if run_id is None:
        raise ValueError(f"Cannot find Wandb ID for pretraining run for env_name={env_name}, use_rep={use_rep}, rep_type={rep_type}, seed={seed}")
    
    run_id = f"{wandb_entity}/{wandb_project}/{run_id}"
    wandb_api = wandb.Api()
    run = wandb_api.run(run_id)
    workdir = run.config['workdir']

    checkpoint_file = Path(base_path).joinpath(workdir).joinpath('wandb/latest-run/files/checkpoints/params_final.pkl')
    if checkpoint_file.exists():
        logger.info(f"Found pretrained checkpoint at {checkpoint_file}")
        return checkpoint_file
    else:
        logger.info(f"Downloading pretrained checkpoint from Wandb to {checkpoint_file}")
        run.file('checkpoints/params_final.pkl').download(
            root=Path(base_path).joinpath(workdir).joinpath('wandb/latest-run/files'),
            replace=False)
        assert checkpoint_file.exists()
        logger.info(f"Successfully downloaded pretrained checkpoint from Wandb to {checkpoint_file}")
        return checkpoint_file
