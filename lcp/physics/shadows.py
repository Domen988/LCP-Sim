
import numpy as np
from shapely.geometry import Polygon
from typing import List, Tuple, Optional
from lcp.core.geometry import PanelGeometry

class ShadowEngine:
    def __init__(self, geo: PanelGeometry):
        self.geo = geo
        # Pre-compute local panel polygon (centered at 0,0)
        # Assuming PanelGeometry gives width (X) and length (Y)
        w = self.geo.width
        l = self.geo.length
        # Vertices in local frame (assuming Z=0 is plane)
        # Order: TopRight, BottomRight, BottomLeft, TopLeft (CCW)
        self.local_coords = np.array([
            [w/2, l/2, 0],
            [w/2, -l/2, 0],
            [-w/2, -l/2, 0],
            [-w/2, l/2, 0]
        ])

    def _get_world_polygon(self, pos: np.ndarray, rot: np.ndarray) -> Polygon:
        """
        Transforms local panel coords to world frame.
        pos: [x,y,z] center
        rot: 3x3 rotation matrix
        """
        # World = Pos + Rot @ Local
        # (4,3) = (4,3) @ (3,3).T + (3,)
        world_verts = (rot @ self.local_coords.T).T + pos
        # Return Shapely Polygon (ignore Z for checking? No, we need 3D for projection)
        # Shapely is 2D. We will project 3D points to 3D plane, then simplify?
        # Actually Shapely Polygon can hold 3D coords but operations are 2D (XY plane).
        # We need custom projection logic anyway.
        return world_verts

    def calculate_loss(self, 
                       target_pos: np.ndarray, 
                       target_rot: np.ndarray,
                       neighbor_data: List[Tuple[np.ndarray, np.ndarray]], 
                       sun_vec: np.ndarray) -> Tuple[float, List[np.ndarray]]:
        """
        Calculates shaded fraction and returns world-space shadow polygons.
        sun_vec: Vector pointing TO the sun (normalized).
        Returns: (fraction, [poly_verts_world, ...])
        """
        # 1. Define Receiver Plane
        C = target_pos
        n = target_rot[:, 2]
        
        # Check if Sun is behind the panel
        if np.dot(n, sun_vec) <= 0:
            return 0.0, []

        # 2. Setup Basis for 2D Clipping
        receiver_poly_2d = Polygon([(p[0], p[1]) for p in self.local_coords])
        total_area = receiver_poly_2d.area
        if total_area <= 1e-6:
            return 0.0, []

        shadow_polys = []

        # 3. Project Neighbors
        for (n_pos, n_rot) in neighbor_data:
            # Check if neighbor is "upstream" (closer to sun)
            d_rn = n_pos - target_pos
            if np.dot(d_rn, sun_vec) <= 0:
                continue

            # Get Neighbor World Vertices
            n_verts_world = (n_rot @ self.local_coords.T).T + n_pos
            
            # Project onto Receiver Plane
            dots_num = (n_verts_world - C) @ n
            dot_denom = np.dot(sun_vec, n)
            
            if abs(dot_denom) < 1e-6:
                continue 
                
            t_vals = dots_num / dot_denom
            projected_world = n_verts_world - np.outer(t_vals, sun_vec)
            
            # Convert to Local 2D
            p_local_3d = (projected_world - C) @ target_rot
            p_local_2d = p_local_3d[:, :2]
            
            # Create Polygon
            poly_n = Polygon(p_local_2d)
            if not poly_n.is_valid:
                poly_n = poly_n.buffer(0)
                
            shadow_polys.append(poly_n)

        if not shadow_polys:
            return 0.0, []

        # 4. Clip with Receiver
        try:
            from shapely.ops import unary_union
            combined_shadow = unary_union(shadow_polys)
            final_shadow = receiver_poly_2d.intersection(combined_shadow)
            
            shrouded_area = final_shadow.area
            fraction = min(1.0, max(0.0, shrouded_area / total_area))
            
            # Convert back to World Polygons for Visualization
            world_polys_out = []
            if not final_shadow.is_empty:
                geoms = [final_shadow] if final_shadow.geom_type == 'Polygon' else final_shadow.geoms
                for g in geoms:
                    if g.is_empty: continue
                    # Local 2D (N, 2)
                    l2d = np.array(g.exterior.coords)
                    # Local 3D (N, 3) with Z=0
                    l3d = np.column_stack([l2d, np.zeros(len(l2d))])
                    # World
                    w3d = (target_rot @ l3d.T).T + target_pos
                    world_polys_out.append(w3d)
            
            return fraction, world_polys_out
            
        except Exception as e:
            print(f"Shadow Error: {e}")
            return 0.0, []
