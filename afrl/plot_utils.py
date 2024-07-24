import jax
import jax.numpy as jnp


def get_traj_v(agent, trajectory):
    @jax.jit
    def get_v(s):
        v1, v2 = agent.network(s, method='value', seed=jax.random.PRNGKey(0))
        return (v1 + v2) / 2
    observations = trajectory['observations']
    all_values = get_v(observations)
    return {
        'value_preds': all_values[..., jnp.newaxis]
    }


def hiql_get_traj_v(agent, trajectory):
    @jax.jit
    def get_v(s, g):
        v1, v2 = agent.network(jax.tree_map(lambda x: x[None], s), jax.tree_map(lambda x: x[None], g), method='value', seed=jax.random.PRNGKey(0))
        return (v1 + v2) / 2
    observations = trajectory['observations']
    all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(observations, observations)
    return {
        'dist_to_beginning': all_values[:, 0],
        'dist_to_end': all_values[:, -1],
        'dist_to_middle': all_values[:, all_values.shape[1] // 2],
    }
