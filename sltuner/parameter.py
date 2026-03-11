from enum import Enum
from typing import Tuple
from dataclasses import dataclass
import numpy as np

class Transform(Enum):
    LINEAR = "linear"
    LOG = "log"

class SampleMethod(Enum):
    UNIFORM = "uniform"
    GRID = "grid"

@dataclass
class ParameterSpec:
    name: str
    low: float
    high: float
    transform: Transform | None = None
    sample_method: SampleMethod = SampleMethod.GRID

    def _infer_transform(self) -> Transform:
        if self.low > 0 and self.high / self.low >= 100:
            return Transform.LOG
        return Transform.LINEAR

    def __post_init__(self):
        if self.transform is None:
            self.transform = self._infer_transform()

        if self.low >= self.high:
            raise NotImplementedError

        if self.transform == Transform.LOG and self.low <= 0:
            raise NotImplementedError

    def sample(self, rng: np.random.Generator, num: int) -> np.ndarray:
        """Draw n independent samples in internal space."""
        match self.sample_method:
            case SampleMethod.UNIFORM:
                if self.transform is Transform.LOG:
                    return rng.uniform(np.log(self.low), np.log(self.high), size=num)
                return rng.uniform(self.low, self.high, size=num)
            case SampleMethod.GRID:
                if self.transform is Transform.LOG:
                    return np.linspace(np.log(self.low), np.log(self.high), num=num)
                return np.linspace(self.low, self.high, num=num)

        raise NotImplementedError

    def to_internal(self, theta: np.ndarray, jacobian: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Transform from external to internal space. Internal = log(theta) for LOG, theta for LINEAR."""
        if self.transform is Transform.LOG:
            phi = np.log(theta)
            # d(log theta)/d(theta) = 1/theta, so jacobian_internal = jacobian * theta
            shape = (-1,) + (1,) * (jacobian.ndim - 1)
            jacobian_internal = jacobian * theta.reshape(shape)
        else:
            phi = theta
            jacobian_internal = jacobian

        return phi, jacobian_internal

    def internal_bounds(self) -> Tuple[float, float]:
        """Return (low, high) in internal space."""
        if self.transform is Transform.LOG:
            return np.log(self.low), np.log(self.high)
        return self.low, self.high

    def to_external(self, theta: np.ndarray) -> np.ndarray:
        """Transform from internal to external space. Reverses to_internal."""
        if self.transform is Transform.LOG:
            return np.exp(theta)
        return theta
