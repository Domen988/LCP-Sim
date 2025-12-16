import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple
from lcp.core.geometry import PanelGeometry
from lcp.physics.engine import PanelState

from lcp.app.theme import Theme

class PlantVisualizer:
    def __init__(self, geometry: PanelGeometry):
        self.geo = geometry
        
    def _create_box_mesh(self, position: np.ndarray, rotation: np.ndarray, color='blue', opacity=1.0) -> go.Mesh3d:
        """
        Creates a Mesh3d for a panel at a given position/rotation.
        """
        # Dimensions
        w = self.geo.width
        l = self.geo.length
        t = self.geo.thickness
        
        # Local Vertices of a Box centered at (0,0,0) (Geometric Center)
        # 8 corners
        x = [-w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2, w/2]
        y = [-l/2, l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2]
        z = [-t/2, -t/2, -t/2, -t/2, t/2, t/2, t/2, t/2]
        
        vertices = np.vstack([x, y, z]) # 3x8
        
        # Apply Logic:
        # Pivot is at `position` (Global).
        # Panel Geom Center is at `pivot_offset` (Local) from Pivot.
        # But `pivot_offset` vector is defined in Local Coords relative to Panel Frame?
        # Spec: "vector from the Mechanical Pivot to the Geometric Center of the panel box."
        # So Center_Local = Pivot_Local + Offset.
        # Pivot_Local is Origin of rotation frame (0,0,0).
        # So Center_Local = Offset.
        # So we shift vertices by Offset first.
        
        ox, oy, oz = self.geo.pivot_offset
        vertices[0,:] += ox
        vertices[1,:] += oy
        vertices[2,:] += oz
        
        # Rotate (Local -> Global)
        # Global = R * Local
        rotated_verts = rotation @ vertices
        
        # Translate to Pivot Position
        rotated_verts[0,:] += position[0]
        rotated_verts[1,:] += position[1]
        rotated_verts[2,:] += position[2]
        
        # Define Mesh Faces (Indices)
        # 0: -x, -y, -z
        # 1: -x, +y, -z
        # 2: +x, +y, -z
        # 3: +x, -y, -z
        # 4: -x, -y, +z
        # 5: -x, +y, +z
        # 6: +x, +y, +z
        # 7: +x, -y, +z
        
        # Plotly Mesh3d uses i, j, k lists of triangles.
        # Box has 12 triangles (2 per face * 6 faces).
        # Let's brute force the indices or use a standard cube pattern.
        # Bottom: 0-1-2, 0-2-3
        # Top: 4-5-6, 4-6-7
        # Front: 0-1-5, 0-5-4
        # Back: 3-2-6, 3-6-7
        # Left: 0-3-7, 0-7-4
        # Right: 1-2-6, 1-6-5
        # Wait, direction matters for normals.
        # I'll just use explicit list.
        
        i = [0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 1, 1]
        j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6]
        k = [2, 3, 6, 7, 5, 4, 6, 7, 7, 4, 6, 5]
        
        return go.Mesh3d(
            x=rotated_verts[0,:],
            y=rotated_verts[1,:],
            z=rotated_verts[2,:],
            i=i, j=j, k=k,
            color=color,
            opacity=opacity,
            flatshading=True,
            name="Panel"
        )
        
    def _get_static_traces(self) -> List[go.Trace]:
        """Returns list of static traces (Ground, North Arrow)."""
        data = []
        
        # 1. Ground Plane (Removed as per user request)
        # g_size = 10.0 
        # data.append(...)

        
        # 2. North Arrow
        theta = np.radians(-5.0)
        arrow_len = 2.0
        ax = arrow_len * np.sin(theta)
        ay = arrow_len * np.cos(theta)
        
        data.append(go.Scatter3d(
            x=[0, ax], y=[0, ay], z=[0.1, 0.1],
            mode='lines+text', text=["", "N"], textposition="top center",
            line=dict(color=Theme.NORTH_ARROW, width=5), showlegend=False, hoverinfo='skip'
        ))
        data.append(go.Cone(
            x=[ax], y=[ay], z=[0.1],
            u=[np.sin(theta)], v=[np.cos(theta)], w=[0],
            sizemode="absolute", sizeref=0.5, anchor="tail",
            colorscale=[[0, Theme.NORTH_ARROW], [1, Theme.NORTH_ARROW]], showscale=False, hoverinfo='skip'
        ))
        
        return data

    def _get_dynamic_traces(self, states: List[PanelState], sun_vec: np.ndarray, show_rays: bool = False, show_pivots: bool = True, show_clash_emphasis: bool = True) -> List[go.Trace]:
        """Returns list of dynamic traces. Optimized & Vectorized."""
        data = []
        
        w = self.geo.width
        l = self.geo.length
        t = self.geo.thickness
        z_face = t / 2.0
        ox, oy, oz = self.geo.pivot_offset
        ray_len = 1.5 * l
        
        # Batch Lists
        p_x, p_y, p_z = [], [], []
        p_i, p_j, p_k = [], [], []
        p_facecolor = []
        p_vert_count = 0
        
        # Shadows (Mesh)
        sh_x, sh_y, sh_z = [], [], []
        sh_i, sh_j, sh_k = [], [], []
        sh_vert_count = 0
        
        pv_x, pv_y, pv_z = [], [], []
        pv_i, pv_j, pv_k = [], [], []
        pv_vert_count = 0
        
        # Pivot Icosahedron (Sphere)
        pivot_rad = 0.1 # 0.2m diam
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
        ])
        # Normalize and Scale
        ico_verts *= pivot_rad / np.linalg.norm(ico_verts[0])
        
        # Icosahedron Indices (20 faces)
        ico_i = [0, 0, 0, 0, 0, 1, 5, 11, 10, 7, 3, 3, 3, 3, 3, 4, 2, 6, 8, 9]
        ico_j = [11, 5, 1, 7, 10, 5, 11, 10, 7, 1, 9, 4, 2, 6, 8, 9, 4, 2, 6, 8]
        ico_k = [5, 1, 7, 10, 11, 9, 4, 2, 6, 8, 4, 2, 6, 8, 9, 5, 11, 10, 7, 1]

        e_x, e_y, e_z = [], [], []
        
        in_x, in_y, in_z = [], [], []
        out_x, out_y, out_z = [], [], []
        # Removed shad_x lists (Shadow visualization disabled)
        
        # --- VECTORIZED IMPLEMENTATION ---
        
        n_panels = len(states)
        if n_panels == 0:
            return []

        # 1. State Arrays
        pos_arr = np.array([s.position for s in states]) # (N, 3)
        rot_arr = np.array([s.rotation for s in states]) # (N, 3, 3)
        
        # Colors
        c_glass_list = []
        for s in states:
             c = Theme.PANEL_GLASS
             if show_clash_emphasis:
                 if s.mode == "STOW": c = Theme.ERROR
                 if s.collision and s.mode == "TRACKING": c = Theme.WARNING
             c_glass_list.append(c)
        
        # 2. Geometry Template (8 vertices)
        dx, dy, dz = w/2.0, l/2.0, t/2.0
        
        corners_local = np.array([
            [-dx, -dy, oz-dz], [dx, -dy, oz-dz], [dx, dy, oz-dz], [-dx, dy, oz-dz], 
            [-dx, -dy, oz+dz], [dx, -dy, oz+dz], [dx, dy, oz+dz], [-dx, dy, oz+dz]
        ]) # (8, 3)
        
        # 3. Batch Transform
        # Global = Rot @ Local.T + Pos
        # numpy matmul broadcasts: (N, 3, 3) @ (3, 8) -> (N, 3, 8)
        
        transformed = (rot_arr @ corners_local.T).transpose(0, 2, 1) + pos_arr[:, np.newaxis, :]
        # transformed shape: (N, 8, 3)
        
        # Flatten for Mesh3d
        all_verts = transformed.reshape(-1, 3)
        p_x = all_verts[:, 0]
        p_y = all_verts[:, 1]
        p_z = all_verts[:, 2]
        
        # 4. Indices
        # Box topology (12 triangles)
        # Vertices 0-7 per panel
        base_i = np.array([4, 4,  0, 0,  0, 0,  1, 1,  2, 2,  3, 3])
        base_j = np.array([5, 6,  2, 3,  1, 5,  2, 6,  3, 7,  0, 4])
        base_k = np.array([6, 7,  1, 2,  5, 4,  6, 5,  7, 6,  4, 7])
        
        # Offset per panel: 0, 8, 16...
        panel_offsets = np.arange(n_panels) * 8
        # (N, 12) + (N, 1) -> (N, 12)
        
        all_i = (base_i[np.newaxis, :] + panel_offsets[:, np.newaxis]).flatten()
        all_j = (base_j[np.newaxis, :] + panel_offsets[:, np.newaxis]).flatten()
        all_k = (base_k[np.newaxis, :] + panel_offsets[:, np.newaxis]).flatten()
        
        # 5. Colors (Facecolor)
        c_side = Theme.PANEL_FRAME
        
        # Construct color array
        # (N, 12)
        face_colors = []
        for c in c_glass_list:
            # 2 glass, 10 side
            face_colors.extend([c]*2 + [c_side]*10)
            
        p_facecolor = face_colors
        
        # --- EDGES (Top Edge 4-5) ---
        # (N, 2, 3) -> indices 4 and 5 from transformed
        p4 = transformed[:, 4, :] # (N, 3)
        p5 = transformed[:, 5, :] # (N, 3)
        
        # Stack for plotting safely [x1, x2, None...]
        nan_arr = np.full((n_panels, 1), None)
        
        e_x = np.column_stack((p4[:,0], p5[:,0], nan_arr)).flatten()
        e_y = np.column_stack((p4[:,1], p5[:,1], nan_arr)).flatten()
        e_z = np.column_stack((p4[:,2], p5[:,2], nan_arr)).flatten()
        
        # --- PIVOTS ---
        if show_pivots:
            # (N, 1, 3) + (1, 12, 3) -> (N, 12, 3)
            piv_transformed = pos_arr[:, np.newaxis, :] + ico_verts[np.newaxis, :, :]
            piv_flat = piv_transformed.reshape(-1, 3)
            pv_x = piv_flat[:, 0]
            pv_y = piv_flat[:, 1]
            pv_z = piv_flat[:, 2]
            
            piv_offsets = np.arange(n_panels) * 12
            base_ico_i = np.array(ico_i)
            base_ico_j = np.array(ico_j)
            base_ico_k = np.array(ico_k)
            
            pv_i = (base_ico_i[np.newaxis, :] + piv_offsets[:, np.newaxis]).flatten()
            pv_j = (base_ico_j[np.newaxis, :] + piv_offsets[:, np.newaxis]).flatten()
            pv_k = (base_ico_k[np.newaxis, :] + piv_offsets[:, np.newaxis]).flatten()
            
        else:
            pv_x, pv_y, pv_z = [], [], []
            pv_i, pv_j, pv_k = [], [], []
        
        # --- SHADOWS & RAYS (Existing Loop logic preserved for shadows) ---
        # Shadows (Mesh)
        sh_x, sh_y, sh_z = [], [], []
        sh_i, sh_j, sh_k = [], [], []
        sh_vert_count = 0
        
        in_x, in_y, in_z = [], [], []
        out_x, out_y, out_z = [], [], []
        
        for s in states:
            # Exact Shadow Polygons
            if s.shadow_polys:
                shift_vec = s.rotation @ (np.array(self.geo.pivot_offset) + np.array([0, 0, t/2.0 + 0.005]))
                for poly in s.shadow_polys:
                    if len(poly) < 3: continue
                    shifted = poly + shift_vec
                    start_idx = sh_vert_count
                    sh_x.extend(shifted[:,0])
                    sh_y.extend(shifted[:,1])
                    sh_z.extend(shifted[:,2])
                    n_v = len(shifted)
                    for k in range(1, n_v - 1):
                        sh_i.append(start_idx)
                        sh_j.append(start_idx + k)
                        sh_k.append(start_idx + k + 1)
                    sh_vert_count += n_v

            if show_rays:
                xs = np.linspace(-w*0.4, w*0.4, 3)
                ys = np.linspace(-l*0.4, l*0.4, 3)
                r_locs = np.array([[lx + ox, ly + oy, z_face + oz] for lx in xs for ly in ys])
                
                r_glob = (s.rotation @ r_locs.T).T + s.position
                r_sun = r_glob + (sun_vec * ray_len)
                r_gnd = r_glob - (sun_vec * ray_len)
                
                # Manual extend is fine for debug rays
                for ii in range(len(r_glob)):
                    in_x.extend([r_sun[ii,0], r_glob[ii,0], None])
                    in_y.extend([r_sun[ii,1], r_glob[ii,1], None])
                    in_z.extend([r_sun[ii,2], r_glob[ii,2], None])
                    out_x.extend([r_glob[ii,0], r_gnd[ii,0], None])
                    out_y.extend([r_glob[ii,1], r_gnd[ii,1], None])
                    out_z.extend([r_glob[ii,2], r_gnd[ii,2], None])

        # Assign back to variables used in TRACES section
        p_i, p_j, p_k = all_i, all_j, all_k
 
        # --- TRACES ---
        
        data.append(go.Mesh3d(
            x=p_x, y=p_y, z=p_z, i=p_i, j=p_j, k=p_k,
            facecolor=p_facecolor, opacity=1.0, flatshading=True,
            showlegend=False, hoverinfo='skip', lightposition=dict(x=0,y=0,z=100)
        ))
        
        # Shadow Trace
        if sh_x:
            data.append(go.Mesh3d(
                x=sh_x, y=sh_y, z=sh_z, i=sh_i, j=sh_j, k=sh_k,
                color=Theme.SHADOW, opacity=1.0, # User Request: Solid dark shadows
                flatshading=True, hoverinfo='skip'
            ))
        
        data.append(go.Scatter3d(
            x=e_x, y=e_y, z=e_z, mode='lines', 
            line=dict(color=Theme.PANEL_EDGE_TOP, width=5), showlegend=False, hoverinfo='skip'
        ))
        
        if show_pivots and len(pv_x) > 0:
            data.append(go.Mesh3d(
                x=pv_x, y=pv_y, z=pv_z, i=pv_i, j=pv_j, k=pv_k,
                color=Theme.PIVOT, opacity=1.0, flatshading=True,
                showlegend=False, hoverinfo='skip'
            ))
            
        if show_rays:
             data.append(go.Scatter3d(x=in_x, y=in_y, z=in_z, mode='lines', line=dict(color='yellow', width=1), showlegend=False, hoverinfo='skip'))
             data.append(go.Scatter3d(x=out_x, y=out_y, z=out_z, mode='lines', line=dict(color='gray', width=10), showlegend=False, hoverinfo='skip'))
                 
        return data

    def render_scene(self, states: List[PanelState], sun_vec: np.ndarray, show_rays: bool = False, show_pivots: bool = True, show_clash_emphasis: bool = True) -> go.Figure:
        """Wrapper for backward compatibility."""
        static = self._get_static_traces()
        dynamic = self._get_dynamic_traces(states, sun_vec, show_rays, show_pivots, show_clash_emphasis)
        
        layout = go.Layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=static + dynamic, layout=layout)
        return fig
