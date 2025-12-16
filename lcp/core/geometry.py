from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class PanelGeometry:
    """
    Defines the physical dimensions of a single panel unit.
    The pivot is assumed to be at the geometric center of the projection unless offset.
    """
    width: float = 1.46
    length: float = 1.46
    thickness: float = 0.15
    # Vector from Pivot Point to Geometric Center of the box (local coords)
    pivot_offset: Tuple[float, float, float] = (0.0, 0.0, -0.38) 
