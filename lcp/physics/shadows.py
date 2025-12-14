from typing import Protocol, List
import numpy as np

class ShadowCalculator(Protocol):
    def calculate_loss(self, 
                       target_idx: int, 
                       neighbors: List[int],
                       sun_vector: np.ndarray,
                       geometry_data: dict) -> float:
        """
        Calculates the fraction of the panel area that is shaded.
        Returns 0.0 (no shadow) to 1.0 (fully shaded).
        """
        ...

class NoShadow:
    def calculate_loss(self, target_idx: int, neighbors: List[int], sun_vector: np.ndarray, geometry_data: dict) -> float:
        return 0.0
