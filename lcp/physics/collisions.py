import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from lcp.core.geometry import PanelGeometry
from lcp.core.config import ScenarioConfig

@dataclass
class CollisionResult:
    is_collision: bool
    collider_id: str = "" # "Row-Col"

class CollisionDetector:
    """
    Implements the 'Double Box Method' for efficient collision detection
    between parallel tracking panels.
    """
    def __init__(self, geometry: PanelGeometry, config: ScenarioConfig):
        self.geo = geometry
        self.cfg = config
        
        # Pre-calculate bounds
        self.half_w = geometry.width / 2.0
        self.half_l = geometry.length / 2.0
        self.half_t = geometry.thickness / 2.0
        
        # Safety envelope
        self.safe_w = self.half_w + config.tolerance
        self.safe_l = self.half_l + config.tolerance
        self.safe_t = self.half_t + config.tolerance

    def check_clash(self, 
                    pivot_a: np.ndarray, 
                    pivot_b: np.ndarray, 
                    rotation_matrix: np.ndarray) -> bool:
        """
        Checks if Panel B clashes with Panel A, assuming both have same orientation.
        
        Args:
            pivot_a: (3,) Global position of pivot A
            pivot_b: (3,) Global position of pivot B
            rotation_matrix: (3,3) Orientation of both panels (Local -> Global)
        """
        # Vector D in Global Frame
        global_d = pivot_b - pivot_a
        
        # Rotate D into Local Frame of Panel A
        # Local = R.T * Global
        local_d = rotation_matrix.T @ global_d
        
        # Check against bounding box
        dx, dy, dz = np.abs(local_d)
        
        # Since we are checking if CENTER of B is inside EXPANDED bounds of A?
        # The spec says: "Check if the center of a Neighbor Panel lies inside the 'Expanded Bounds' of the Target Panel."
        # Because panels are parallel, the bounding box of B is axis-aligned with A.
        # The condition for intersection of two AABBs (Axis Aligned Bounding Boxes):
        # |center_dist_x| < (half_size_a + half_size_b)
        # Here sizes are identical. So |dx| < (half_w + half_w) = width.
        # Plus tolerance.
        
        # Spec Logic: "If |dx_local| < (Width + Tolerance)..."
        # My geom has Width = full width.
        # So we check if center distance < Width. 
        # Yes, |x1 - x2| < w1/2 + w2/2 = w.
        
        limit_x = self.geo.width + self.cfg.tolerance
        limit_y = self.geo.length + self.cfg.tolerance
        limit_z = self.geo.thickness + self.cfg.tolerance
        
        if (dx < limit_x) and (dy < limit_y) and (dz < limit_z):
            return True
            
        return False
