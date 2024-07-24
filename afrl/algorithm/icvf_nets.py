from typing import Sequence, Dict, Callable

import jax.numpy as jnp
import flax.linen as nn

from jaxrl_m.typing import PRNGKey, Shape, Dtype, Array
from jaxrl_m.networks import MLP, get_latent, ensemblize, default_init


class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: bool = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:                
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x


class MultilinearVF(nn.Module):
    hidden_dims: Sequence[int]
    use_layer_norm: bool = False

    def setup(self):
        network_cls = LayerNormMLP if self.use_layer_norm else MLP
        self.phi_net = network_cls(self.hidden_dims, activate_final=True, name='phi')
        self.psi_net = network_cls(self.hidden_dims, activate_final=True, name='psi')

        self.T_net =  network_cls(self.hidden_dims, activate_final=True, name='T')

        self.matrix_a = nn.Dense(self.hidden_dims[-1], name='matrix_a')
        self.matrix_b = nn.Dense(self.hidden_dims[-1], name='matrix_b')
        
    
    def __call__(self, observations: jnp.ndarray, outcomes: jnp.ndarray, intents: jnp.ndarray) -> jnp.ndarray:
        return self.get_info(observations, outcomes, intents)['v']
        
    def get_phi(self, observations):
        return self.phi_net(observations)
    
    def get_psi(self, outcomes):
        return self.psi_net(outcomes)

    def get_info(self, observations: jnp.ndarray, outcomes: jnp.ndarray, intents: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        phi = self.phi_net(observations)
        psi = self.psi_net(outcomes)
        z = self.psi_net(intents)
        Tz = self.T_net(z)

        # T(z) should be a dxd matrix, but having a network output d^2 parameters is inefficient
        # So we'll make a low-rank approximation to T(z) = (diag(Tz) * A * B * diag(Tz))
        # where A and B are (fixed) dxd matrices and Tz is a d-dimensional parameter dependent on z

        phi_z = self.matrix_a(Tz * phi)
        psi_z = self.matrix_b(Tz * psi)
        v = (phi_z * psi_z).sum(axis=-1)

        return {
            'v': v,
            'phi': phi,
            'psi': psi,
            'Tz': Tz,
            'z': z,
            'phi_z': phi_z,
            'psi_z': psi_z,
        }


class Connector(nn.Module):
    rep_dim: int
    use_layer_norm: bool = False
    
    def setup(self):
        network_cls = LayerNormMLP if self.use_layer_norm else MLP
        self.connector_net = network_cls([self.rep_dim, ], activate_final=True, name='connector')
    
    def __call__(self, x):
        return self.connector_net(x)


def create_icvf(rep_dim, hidden_dims, use_layer_norm, create_connector: bool):
    vf = ensemblize(MultilinearVF, 2, methods=['__call__', 'get_info', 'get_phi', 'get_psi'])(hidden_dims=hidden_dims, use_layer_norm=use_layer_norm)
    if not create_connector:
        return vf
    else:
        connector = Connector(rep_dim=rep_dim, use_layer_norm=use_layer_norm)
        return vf, connector
