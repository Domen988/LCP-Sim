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
            return self._check_sat(pivot_a, pivot_b, rot_a, rot_b)

    def _check_parallel(self, pivot_a, pivot_b, rotation_matrix):
        # ... (unchanged part implicit, but I need to replace content if I use replace_file_content)
        # Wait, I should keep _check_parallel as is.
        # But for replace_file_content on the whole block, I need to include it or just replace the methods.
        # Let's just implement _check_parallel again to be safe or use the tool to replace ONLY from check_clash down?
        # The prompt asks to rewrite _check_sat.
        # I will replace check_clash and below.
        
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
        Vectorized for (N, 3, 3) input.
        """
        is_vector = (rot_a.ndim == 3)
        
        offset = np.array(self.geo.pivot_offset)
        
        # 1. Setup Box A
        if is_vector:
            # rot_a (N, 3, 3) @ offset (3,) -> (N, 3)
            c_a = pivot_a + np.matmul(rot_a, offset)
            # Axes: Columns of rot matrix. 
            # (N, 3, 3) -> col 0 is vector slice [:, :, 0] shape (N, 3)
            ax_a = [rot_a[:,:,0], rot_a[:,:,1], rot_a[:,:,2]]
        else:
            c_a = pivot_a + rot_a @ offset
            ax_a = [rot_a[:,0], rot_a[:,1], rot_a[:,2]]
        
        # Extents
        tol_half = self.cfg.tolerance / 2.0
        # W/L/T map to x/y/z in local frame? 
        # Usually Panel Geometry is X=Width, Y=Length, Z=Thickness
        ext_arr = np.array([
            self.geo.width/2.0 + tol_half,
            self.geo.length/2.0 + tol_half,
            self.geo.thickness/2.0 + tol_half
        ])
        
        # 2. Setup Box B
        if is_vector:
            c_b = pivot_b + np.matmul(rot_b, offset)
            ax_b = [rot_b[:,:,0], rot_b[:,:,1], rot_b[:,:,2]]
        else:
            c_b = pivot_b + rot_b @ offset
            ax_b = [rot_b[:,0], rot_b[:,1], rot_b[:,2]]
            
        ext_b = ext_arr # Same geometry
        ext_a = ext_arr

        # 3. Translation Vector
        T = c_b - c_a # (N, 3) or (3,)

        # Initialize Result (Assumption: Colliding until Separated)
        if is_vector:
            colliding = np.ones(len(rot_a), dtype=bool)
        else:
            colliding = True

        # Helper for vectorized dot product
        def vdot(v1, v2):
            # v1 (N,3), v2 (N,3) -> dot -> (N,)
            if is_vector:
                return np.sum(v1 * v2, axis=1)
            else:
                return np.dot(v1, v2)

        # Helper Check Loop
        # 15 Axes total
        
        axes_to_check = []
        # 3 Face Normals A
        axes_to_check.extend(ax_a)
        # 3 Face Normals B
        axes_to_check.extend(ax_b)
        
        # 9 Cross Products
        for i in range(3):
            for j in range(3):
                if is_vector:
                    # (N,3) x (N,3) -> (N,3)
                    axis = np.cross(ax_a[i], ax_b[j])
                    # Normalize ?? 
                    # SAT works with non-normalized axes but projection magnitudes scale?
                    # "L = |T . L| > sum |Proj a| + |Proj b|"
                    # Proj = rad * |unit_axis . L| -> if L non-unit, then dot scales L.
                    # Formula is consistent if we don't normalize, AS LONG AS we compare projected T vs projected radii?
                    # Wait. Ref: "Standard SAT" uses Unit axes.
                    # If axis is small (parallel cross), we skip.
                    
                    # Norm check vectorized:
                    norms = np.linalg.norm(axis, axis=1)
                    # Avoid division by zero
                    # If norm < epsilon, skip that index? 
                    # Or just set axis to 0 (which yields 0 projection -> no separation found on this axis -> continue)
                    valid = norms > 1e-6
                    
                    # We only need to check separation where valid.
                    # But simpler: Normalize valid ones, others 0.
                    # np.divide where
                    axis = np.divide(axis, norms[:,None], out=np.zeros_like(axis), where=norms[:,None]>1e-6)
                    axes_to_check.append(axis)
                else:
                    axis = np.cross(ax_a[i], ax_b[j])
                    if np.dot(axis, axis) > 1e-6:
                        axis = axis / np.linalg.norm(axis)
                        axes_to_check.append(axis)

        # Check Overlap on all axes
        # If ANY axis separates -> Not Colliding (False)
        
        for axis in axes_to_check:
            # If we already found separation for all items (colliding all False), break?
            if is_vector:
                if not np.any(colliding): 
                    break
                    
                # Calculate Projections
                t_proj = np.abs(vdot(T, axis))
                
                ra = (ext_a[0] * np.abs(vdot(ax_a[0], axis)) +
                      ext_a[1] * np.abs(vdot(ax_a[1], axis)) +
                      ext_a[2] * np.abs(vdot(ax_a[2], axis)))
                      
                rb = (ext_b[0] * np.abs(vdot(ax_b[0], axis)) +
                      ext_b[1] * np.abs(vdot(ax_b[1], axis)) +
                      ext_b[2] * np.abs(vdot(ax_b[2], axis)))
                      
                # Separation Condition: T_proj > Ra + Rb
                separated = t_proj > (ra + rb)
                
                # Update colliding state: If separated, set False
                colliding = colliding & (~separated)
                
            else:
                if not colliding: break
                
                t_proj = abs(np.dot(T, axis))
                ra = sum(ext_a[k] * abs(np.dot(ax_a[k], axis)) for k in range(3))
                rb = sum(ext_b[k] * abs(np.dot(ax_b[k], axis)) for k in range(3))
                
                if t_proj > (ra + rb):
                    colliding = False

        return colliding

    def _sat_overlap(self, T, axis, ax_a, ext_a, ax_b, ext_b):
        # Deprecated helper in favor of inlined version for easier vectorization context sharing
        pass
