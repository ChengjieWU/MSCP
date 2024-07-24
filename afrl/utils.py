import jax
from jax import numpy as jnp
import flax
import wandb


class CsvLogger:
    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        row['step'] = step
        if self.file is None:
            self.file = open(self.path, 'w')
            if self.header is None:
                self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
                self.file.write(','.join(self.header) + '\n')
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        else:
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def compare_frozen_dicts(a: flax.core.FrozenDict, b: flax.core.FrozenDict):
    assert isinstance(a, flax.core.FrozenDict)
    assert isinstance(b, flax.core.FrozenDict)
    if a.keys() != b.keys():
        return False
    for k in a.keys():
        assert isinstance(a[k], flax.core.FrozenDict) or isinstance(a[k], jax.Array)
        assert isinstance(b[k], flax.core.FrozenDict) or isinstance(b[k], jax.Array)
        if type(a[k]) != type(b[k]):
            return False
        if isinstance(a[k], flax.core.FrozenDict):
            sub_compare = compare_frozen_dicts(a[k], b[k])
        else:
            sub_compare = jnp.allclose(a[k], b[k])
        if not sub_compare:
            return False
    return True
