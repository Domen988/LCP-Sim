import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple
from lcp.core.geometry import PanelGeometry
from lcp.physics.engine import PanelState

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
            line=dict(color='red', width=5), showlegend=False, hoverinfo='skip'
        ))
        data.append(go.Cone(
            x=[ax], y=[ay], z=[0.1],
            u=[np.sin(theta)], v=[np.cos(theta)], w=[0],
            sizemode="absolute", sizeref=0.5, anchor="tail",
            colorscale=[[0, 'red'], [1, 'red']], showscale=False, hoverinfo='skip'
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
        
        # --- LOOP STATES ---
        for s_idx, s in enumerate(states):
            # 1. Colors
            c_glass = '#1f77b4' # Blue
            if show_clash_emphasis:
                if s.mode == "STOW": c_glass = 'red' 
                if s.collision and s.mode == "TRACKING": c_glass = 'orange'
            # Removed gray logic for shadows
            c_side = 'darkgray'
            
            # 2. Panel Geometry
            dx, dy, dz = w/2.0, l/2.0, t/2.0
            corners_loc = np.array([
                [-dx, -dy, oz-dz], [dx, -dy, oz-dz], [dx, dy, oz-dz], [-dx, dy, oz-dz], 
                [-dx, -dy, oz+dz], [dx, -dy, oz+dz], [dx, dy, oz+dz], [-dx, dy, oz+dz]
            ])
            gl_corn = (s.rotation @ corners_loc.T).T + s.position
            
            p_x.extend(gl_corn[:,0])
            p_y.extend(gl_corn[:,1])
            p_z.extend(gl_corn[:,2])
            
            # Panel Indices
            curr_i = [4, 4,  0, 0,  0, 0,  1, 1,  2, 2,  3, 3]
            curr_j = [5, 6,  2, 3,  1, 5,  2, 6,  3, 7,  0, 4]
            curr_k = [6, 7,  1, 2,  5, 4,  6, 5,  7, 6,  4, 7]
            
            p_i.extend([x + p_vert_count for x in curr_i])
            p_j.extend([x + p_vert_count for x in curr_j])
            p_k.extend([x + p_vert_count for x in curr_k])
            
            p_facecolor.extend([c_glass]*2 + [c_side]*10)
            p_vert_count += 8
            
            # 3. Top Edge (User requested swap, using indices 4 and 5)
            p4, p5 = gl_corn[4], gl_corn[5]
            e_x.extend([p4[0], p5[0], None])
            e_y.extend([p4[1], p5[1], None])
            e_z.extend([p4[2], p5[2], None])
            
            # 4. Pivots (Icosahedron)
            if show_pivots:
                pv_gl = ico_verts + s.position
                
                pv_x.extend(pv_gl[:,0])
                pv_y.extend(pv_gl[:,1])
                pv_z.extend(pv_gl[:,2])
                
                pv_i.extend([x + pv_vert_count for x in ico_i])
                pv_j.extend([x + pv_vert_count for x in ico_j])
                pv_k.extend([x + pv_vert_count for x in ico_k])
                
                pv_vert_count += 12
            
            # 5. Exact Shadow Polygons
            if s.shadow_polys:
                # Calculate offset to glass surface
                # Pivot Frame -> Geom Center (Pivot Offset) -> Top Surface (Thickness/2)
                # Ensure we push slightly +0.05 to avoid z-fight with glass
                shift_vec = s.rotation @ (np.array(self.geo.pivot_offset) + np.array([0, 0, t/2.0 + 0.05]))
                
                for poly in s.shadow_polys:
                    if len(poly) < 3: continue
                    
                    shifted = poly + shift_vec
                    
                    # Vertices
                    start_idx = sh_vert_count
                    sh_x.extend(shifted[:,0])
                    sh_y.extend(shifted[:,1])
                    sh_z.extend(shifted[:,2])
                    
                    n_v = len(shifted)
                    # Simple Fan Triangulation (Converions ok for convex-ish)
                    for k in range(1, n_v - 1):
                        sh_i.append(start_idx)
                        sh_j.append(start_idx + k)
                        sh_k.append(start_idx + k + 1)
                    
                    sh_vert_count += n_v

            # 6. Rays (No Intersection Check)
            if show_rays:
                xs = np.linspace(-w*0.4, w*0.4, 3)
                ys = np.linspace(-l*0.4, l*0.4, 3)
                
                r_locs = []
                for lx in xs:
                    for ly in ys:
                        r_locs.append([lx + ox, ly + oy, z_face + oz])
                r_locs = np.array(r_locs)
                
                r_glob = (s.rotation @ r_locs.T).T + s.position
                r_sun = r_glob + (sun_vec * ray_len)
                r_gnd = r_glob - (sun_vec * ray_len)
                
                for ii in range(len(r_glob)):
                    in_x.extend([r_sun[ii,0], r_glob[ii,0], None])
                    in_y.extend([r_sun[ii,1], r_glob[ii,1], None])
                    in_z.extend([r_sun[ii,2], r_glob[ii,2], None])
                    
                    out_x.extend([r_glob[ii,0], r_gnd[ii,0], None])
                    out_y.extend([r_glob[ii,1], r_gnd[ii,1], None])
                    out_z.extend([r_glob[ii,2], r_gnd[ii,2], None])
 
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
                color='black', opacity=1.0, # User Request: Solid dark shadows
                flatshading=True, hoverinfo='skip'
            ))
        
        data.append(go.Scatter3d(
            x=e_x, y=e_y, z=e_z, mode='lines', 
            line=dict(color='black', width=5), showlegend=False, hoverinfo='skip'
        ))
        
        if show_pivots and pv_x:
            data.append(go.Mesh3d(
                x=pv_x, y=pv_y, z=pv_z, i=pv_i, j=pv_j, k=pv_k,
                color='teal', opacity=1.0, flatshading=True,
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
