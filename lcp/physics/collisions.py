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
                    rot_a: np.ndarray,
                    rot_b: np.ndarray = None) -> bool | np.ndarray:
        """
        Checks collision between Panel A and Panel B.
        If rot_b is None, assumes parallel panels (Optimized).
        If rot_b is provided, uses Separating Axis Theorem (SAT).
        Supports vectorized inputs for rot_a (N, 3, 3).
        """
        if rot_b is None:
            return self._check_parallel(pivot_a, pivot_b, rot_a)
        else:
            # SAT Not Vectorized yet
            if rot_a.ndim == 3:
                raise NotImplementedError("Vectorized SAT check not implemented.")
            return self._check_sat(pivot_a, pivot_b, rot_a, rot_b)

    def _check_parallel(self, pivot_a, pivot_b, rotation_matrix):
        # Vector D in Global Frame
        global_d = pivot_b - pivot_a
        
        # Handle Vectorization
        if rotation_matrix.ndim == 3:
            # rotation_matrix shape: (N, 3, 3)
            # Transpose of each matrix: (N, 3, 3) -> swap last two axes
            rot_T = np.swapaxes(rotation_matrix, 1, 2)
            
            # Matmul: (N, 3, 3) @ (3,) -> (N, 3)
            # broadcasting global_d (3,) across N
            local_d = np.matmul(rot_T, global_d)
        else:
            # Scalar case
            local_d = rotation_matrix.T @ global_d
        
        dx, dy, dz = np.abs(local_d.T) # Unpack columns (N, 3) -> (3, N)
        # If scalar, unpacks (3,) -> dx, dy, dz scalars
        
        limit_x = self.geo.width + self.cfg.tolerance
        limit_y = self.geo.length + self.cfg.tolerance
        limit_z = self.geo.thickness + self.cfg.tolerance
        
        # Vectorized comparison
        clash_x = dx < limit_x
        clash_y = dy < limit_y
        clash_z = dz < limit_z
        
        return clash_x & clash_y & clash_z

    def _check_sat(self, pivot_a, pivot_b, rot_a, rot_b):
        """
        General OBB Collision using Separating Axis Theorem (SAT).
        """
        # 1. Setup Box A
        # Center A (Global) = Pivot A + Rot A @ Offset
        # We need "Geometric Center" for SAT symmetry
        c_a = pivot_a + rot_a @ np.array(self.geo.pivot_offset)
        axes_a = rot_a.T # Rows are axes? Global axes are Columns of rot_a.
        # Axes should be the unitary vectors. rot_a columns.
        ax_a = [rot_a[:,0], rot_a[:,1], rot_a[:,2]]
        
        # Extents A (Half-Widths + Tolerance/2)
        # Note: tolerance in parallel check was added to FULL width.
        # So "dist < W + tol".
        # In SAT: "dist < radA + radB". radA = W/2 + tol/2.
        # So sum radii = W + tol. Consistent.
        tol_half = self.cfg.tolerance / 2.0
        ext_a = np.array([
            self.geo.width/2.0 + tol_half,
            self.geo.length/2.0 + tol_half,
            self.geo.thickness/2.0 + tol_half
        ])

        # 2. Setup Box B
        c_b = pivot_b + rot_b @ np.array(self.geo.pivot_offset)
        ax_b = [rot_b[:,0], rot_b[:,1], rot_b[:,2]]
        ext_b = ext_a # Same geometry

        # 3. Translation Vector
        T = c_b - c_a

        # 4. Check Axes
        # 3 Face Normals of A
        for i in range(3):
            if not self._sat_overlap(T, ax_a[i], ax_a, ext_a, ax_b, ext_b): return False
        
        # 3 Face Normals of B
        for i in range(3):
            if not self._sat_overlap(T, ax_b[i], ax_a, ext_a, ax_b, ext_b): return False

        # 9 Cross Products
        for i in range(3):
            for j in range(3):
                # Robust Cross Product
                axis = np.cross(ax_a[i], ax_b[j])
                # If parallel axes, cross product is 0. Skip.
                if np.dot(axis, axis) < 1e-6: continue
                axis = axis / np.linalg.norm(axis)
                if not self._sat_overlap(T, axis, ax_a, ext_a, ax_b, ext_b): return False

        return True

    def _sat_overlap(self, T, axis, ax_a, ext_a, ax_b, ext_b):
        # Project T
        t_proj = abs(np.dot(T, axis))
        
        # Project A
        # Radius A = sum( extent_i * abs(dot(axis_i, axis)) )
        ra = (ext_a[0] * abs(np.dot(ax_a[0], axis)) +
              ext_a[1] * abs(np.dot(ax_a[1], axis)) +
              ext_a[2] * abs(np.dot(ax_a[2], axis)))
              
        # Project B
        rb = (ext_b[0] * abs(np.dot(ax_b[0], axis)) +
              ext_b[1] * abs(np.dot(ax_b[1], axis)) +
              ext_b[2] * abs(np.dot(ax_b[2], axis)))
              
        return t_proj < (ra + rb)
