from dataclasses import dataclass
from typing import Literal
import numpy as np

Backend = Literal["numpy", "jax", "torch"]

@dataclass
class BackendOps:
    backend: Backend
    xp: object

    def asarray(self, x, atleast_2d: bool = False):
        if self.backend == "torch":
            if isinstance(x, self.xp.Tensor):
                arr = x.to(dtype=self.xp.float64)
            else:
                arr = self.xp.as_tensor(x, dtype=self.xp.float64)
            if atleast_2d and arr.ndim < 2:
                if arr.ndim == 1:
                    arr = arr.unsqueeze(0)
                else:
                    arr = arr.reshape(1, 1)
            return arr

        arr = self.xp.asarray(x, dtype=self.xp.float64)
        if atleast_2d:
            arr = self.xp.atleast_2d(arr)
        return arr

    def maximum(self, x, y):
        return self.xp.maximum(x, y)

    def minimum(self, x, y):
        return self.xp.minimum(x, y)

    def clamp(self, x, min=None, max=None):
        if self.backend == "torch":
            return self.xp.clamp(x, min=min, max=max)
        if min is not None:
            x = self.xp.maximum(x, min)
        if max is not None:
            x = self.xp.minimum(x, max)
        return x

    def clamp_min(self, x, min_value):
        return self.maximum(x, min_value)

    def clamp_max(self, x, max_value):
        return self.minimum(x, max_value)

    def exp(self, x):
        return self.xp.exp(x)

def get_ops(backend: Backend) -> BackendOps:
    if backend == "numpy":
        xp = np
    elif backend == "jax":
        import jax.numpy as jnp
        xp = jnp
    elif backend == "torch":
        import torch
        xp = torch
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return BackendOps(backend=backend, xp=xp)