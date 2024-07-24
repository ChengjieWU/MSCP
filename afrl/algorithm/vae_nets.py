from typing import Callable, Dict, Sequence

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn

from jaxrl_m.typing import PRNGKey, Shape, Dtype, Array
from jaxrl_m.networks import MLP, default_init


class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:                
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x


def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


def vae_encode(
        rng: jax.random.PRNGKey, encoder: nn.Module, targets: jnp.ndarray, bases: jnp.ndarray = None,
):
    if bases is None:
        mean, logvar = encoder(targets)
    else:
        mean, logvar = encoder(targets, bases)
    return reparameterize(rng, mean, logvar)


class RelativeVAEEncoder(nn.Module):
    rep_dim: int = 32
    hidden_dims: tuple = (128, 64)
    module: nn.Module = None
    visual: bool = False
    layer_norm: bool = False
    rep_type: str = 'state'

    @nn.compact
    def __call__(self, targets, bases=None):
        if bases is None:
            inputs = targets
        else:
            if self.rep_type == 'state':
                inputs = targets
            elif self.rep_type == 'diff':
                inputs = jax.tree_map(lambda t, b: t - b + jnp.ones_like(t) * 1e-6, targets, bases)
            elif self.rep_type == 'concat':
                inputs = jax.tree_map(lambda t, b: jnp.concatenate([t, b], axis=-1), targets, bases)
            else:
                raise NotImplementedError

        if self.visual:
            raise NotImplementedError
            inputs = self.module()(inputs)
        if self.layer_norm:
            rep = LayerNormMLP(self.hidden_dims, activate_final=True, activations=nn.gelu)(inputs)
        else:
            rep = MLP(self.hidden_dims, activate_final=True, activations=nn.gelu)(inputs)

        mean = nn.Dense(self.rep_dim, name='fc_mean', kernel_init=default_init())(rep)
        logvar = nn.Dense(self.rep_dim, name='fc_logvar', kernel_init=default_init())(rep)

        return mean, logvar


class VAEDecoder(nn.Module):
    original_dim: int = 256
    hidden_dims: tuple = (64, 128)
    module: nn.Module = None
    visual: bool = False
    layer_norm: bool = False

    @nn.compact
    def __call__(self, z):
        if self.visual:
            raise NotImplementedError
            inputs = self.module()(z)
        if self.layer_norm:
            z = LayerNormMLP(self.hidden_dims, activate_final=True, activations=nn.gelu)(z)
        else:
            z = MLP(self.hidden_dims, activate_final=True, activations=nn.gelu)(z)

        z = nn.Dense(self.original_dim, name='fc_output', kernel_init=default_init())(z)

        return z
