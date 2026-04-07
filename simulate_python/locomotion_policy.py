"""Abstract interface for locomotion policies running in MuJoCo."""

from abc import ABC, abstractmethod

import numpy as np


class LocomotionPolicy(ABC):
    """Base class for locomotion policies.

    Subclasses must set: kp, kd, hz, standing_pos (np.ndarray in ctrl order).
    """

    kp: float  # PD position gain
    kd: float  # PD velocity gain
    hz: float  # Policy control frequency (Hz)
    standing_pos: np.ndarray  # Default standing pose in MuJoCo ctrl order

    @abstractmethod
    def step(
        self, sensordata: np.ndarray, vx: float, vy: float, vyaw: float
    ) -> np.ndarray:
        """Run one policy step.

        Returns target joint positions in MuJoCo ctrl order (FR, FL, RR, RL).
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset policy internal state."""
