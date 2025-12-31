
import numpy as np
import pyqtgraph.opengl as gl
import pyqtgraph
from PyQt6.QtCore import pyqtSignal, QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from lcp.gui.state import AppState

class PlantViewport(gl.GLViewWidget):
    """
    The 3D OpenGL Viewport. 
    High-performance rendering with batching.
    Features:
    - Thick Panels (Pastel Blue)
    - Pivots (Red Spheres)
    - Sunrays (9 per panel):
        - Yellow Lines: Panel Center -> Up (Normal)
        - Gray Cylinders: Panel Center -> Ground (Along Negative Normal)
    - Shadows on Panels (Extruded "Prisms" to avoid Z-fighting)
    - North Arrow
    - Smart Parity Mapping (Stow Strategy)
    - State Persistence (Remembers last frame on toggle)
    - Tolerance Visualization (Show enlarged panels)
    """
    
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        
        # State Cache (For toggles)
        self.last_states = None
        self.last_az = None
        self.last_el = None
        self.last_safety = False
        
        # Viz Settings
        self.show_pivots = True
        self.show_rays = False
        self.show_full_plant = False 
        self.show_stow = False 
        self.show_tolerance = False
        self.limit_view = 3 
        
        # Camera
        self.setCameraPosition(distance=120, elevation=45, azimuth=-90)
        self.opts['fov'] = 1 
        self.opts['center'] = pyqtgraph.Vector(0, 0, 0)
        self.setBackgroundColor((50, 50, 50, 255))
        
        # --- SCENE ITEMS ---
        
        # 1. Grid
        g = gl.GLGridItem()
        g.setSize(x=100, y=100)
        g.setSpacing(x=5, y=5)
        g.setColor((100, 100, 100, 255)) 
        self.addItem(g)
        
        # 2. Plant Mesh (Unlit)
        self.plant_mesh = gl.GLMeshItem(
            meshdata=None, 
            smooth=False, 
            glOptions='opaque'
        )
        self.addItem(self.plant_mesh)
        
        # 3. Pivot Mesh (Red Spheres)
        self.pivot_mesh = gl.GLMeshItem(
            meshdata=None,
            smooth=True, 
            color=(0.8, 0.1, 0.1, 1), 
            glOptions='opaque'
        )
        self.addItem(self.pivot_mesh)
        
        # 4. Shadow on Panel Mesh (Thick Prism)
        self.panel_shadow_mesh = gl.GLMeshItem(
            meshdata=None,
            smooth=False,
            color=(0.05, 0.05, 0.05, 1), 
            glOptions='opaque'
        )
        self.addItem(self.panel_shadow_mesh)
        
        # 4a. Panel Edge Mesh (Thick Purple)
        self.edge_mesh = gl.GLMeshItem(
            meshdata=None,
            smooth=False,
            color=(0.6, 0.0, 0.8, 1), 
            glOptions='opaque'
        )
        self.addItem(self.edge_mesh)
        
        # 5. Sunrays
        self.rays_sun = gl.GLLinePlotItem(width=3, color=(1.0, 0.8, 0.0, 1), mode='lines') 
        self.addItem(self.rays_sun)
        
        self.rays_gnd_mesh = gl.GLMeshItem(
            meshdata=None,
            smooth=True,
            color=(0.6, 0.6, 0.6, 0.4), 
            glOptions='translucent'
        )
        self.addItem(self.rays_gnd_mesh)
        
        # 6. North Arrow
        self.north_arrow = gl.GLLinePlotItem(width=5, color=(1.0, 0.2, 0.2, 1), glOptions='opaque')
        self.addItem(self.north_arrow)
        
        self.txt_north = gl.GLTextItem(pos=[0, 0, 0], text='N', color=(1,0.2,0.2,1))
        self.addItem(self.txt_north)
        
        # Base Arrow
        self.arrow_base_pts = np.array([
             [0, 0, 0.2], [0, 5, 0.2],
             [-1, 4, 0.2], [0, 5, 0.2],
             [1, 4, 0.2], [0, 5, 0.2]
        ])
        
        # Precompute
        self.base_verts, self.base_faces = self._create_box_verts(
            self.state.geometry.width, 
            self.state.geometry.length, 
            self.state.geometry.thickness
        )
        
        self.ico_verts, self.ico_faces = self._create_icosahedron(radius=0.075) 
        self.cyl_verts, self.cyl_faces = self._create_unit_cylinder(radius=0.05, segments=6)
        
        # Initial Draw
        self.update_scene()
    
    def fit_to_plant(self):
        """Resets camera to fit the whole plant."""
        # Force full plant view
        self.show_full_plant = True
        
        rows = self.state.rows
        cols = self.state.cols
        px = self.state.config.grid_pitch_x
        py = self.state.config.grid_pitch_y
        
        w = cols * px # Estimate
        h = rows * py # Estimate
        
        # Better estimate with fields
        f_sp_x = getattr(self.state.config, 'field_spacing_x', px)
        f_sp_y = getattr(self.state.config, 'field_spacing_y', py)
        
        if cols > 0:
             lc = cols - 1
             w = (lc // 4) * ((3 * px) + f_sp_x) + (lc % 4) * px
        if rows > 0:
             lr = rows - 1
             h = (lr // 4) * ((3 * py) + f_sp_y) + (lr % 4) * py
        dim = max(w, h)
        
        dist = dim * 1.5 + 50
        
        self.setCameraPosition(distance=dist, elevation=45, azimuth=-90)
        self.opts['center'] = pyqtgraph.Vector(0, 0, 0)
        self.update_scene()
        self.update()

    def _create_box_verts(self, w, l, t):
        dx, dy, dz = w/2, l/2, t/2
        
        verts = np.array([
            [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
            [-dx, -dy, dz],  [dx, -dy, dz],  [dx, dy, dz],  [-dx, dy, dz]
        ])
        
        faces = np.array([
             [0, 1, 2], [0, 2, 3], 
             [4, 5, 6], [4, 6, 7], 
             [0, 1, 5], [0, 5, 4], 
             [1, 2, 6], [1, 6, 5], 
             [2, 3, 7], [2, 7, 6], 
             [3, 0, 4], [3, 4, 7]  
        ])
        return verts, faces
        
    def _create_icosahedron(self, radius):
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
        ])
        verts = verts * radius / np.linalg.norm(verts[0])
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ])
        return verts, faces
        
    def _create_unit_cylinder(self, radius, segments):
        vs = []
        fs = []
        angle = 2 * np.pi / segments
        
        # 0: Bot Center, 1: Top Center
        vs.append([0, 0, 0])
        vs.append([0, 0, 1])
        
        for i in range(segments):
             theta = i * angle
             vs.append([radius*np.cos(theta), radius*np.sin(theta), 0]) # Bot
        for i in range(segments):
             theta = i * angle
             vs.append([radius*np.cos(theta), radius*np.sin(theta), 1]) # Top
             
        # Faces
        for i in range(segments):
            curr = i
            next_ = (i + 1) % segments
            
            # Bottom Cap
            fs.append([0, 2+next_, 2+curr])
            # Top Cap
            fs.append([1, 2+segments+curr, 2+segments+next_])
            # Sides
            cb = 2 + curr
            nb = 2 + next_
            ct = 2 + segments + curr
            nt = 2 + segments + next_
            fs.append([cb, nb, nt])
            fs.append([cb, nt, ct])
            
        return np.array(vs), np.array(fs)

    def update_from_frame(self, sun_az, sun_el, safety, states=None):
        # Cache for toggles
        self.last_az = sun_az
        self.last_el = sun_el
        self.last_safety = safety
        self.last_states = states
        
        self.update_scene(states=states, explicit_az=sun_az, explicit_el=sun_el)
        
    def set_show_pivots(self, val):
        self.show_pivots = val
        self.pivot_mesh.setVisible(val)
        
    def set_show_rays(self, val):
        self.show_rays = val
        self.update_scene()
        
    def set_show_full_plant(self, val):
        self.show_full_plant = val
        self.update_scene()
        
    def set_show_tolerance(self, val):
        self.show_tolerance = val
        self.update_scene()
        
    def update_scene(self, states=None, explicit_az=None, explicit_el=None):
        # Use Cache if args missing (Toggle case)
        if states is None: states = self.last_states
        if explicit_az is None: explicit_az = self.last_az
        if explicit_el is None: explicit_el = self.last_el
        
        s = self.state
        
        if explicit_az is not None:
            eff_sun_az = explicit_az
            eff_sun_el = explicit_el
        else:
            eff_sun_az = s.sun_az
            eff_sun_el = s.sun_el
            
        rows, cols = s.rows, s.cols
        pitch_x = s.config.grid_pitch_x
        pitch_y = s.config.grid_pitch_y
        
        # North Arrow 
        theta_deg = -(-s.config.plant_rotation) # User requested negative rotation usage
        c_a = np.cos(np.radians(theta_deg))
        s_a = np.sin(np.radians(theta_deg))
        
        pts = self.arrow_base_pts.copy()
        x = pts[:,0]
        y = pts[:,1]
        pts[:,0] = x * c_a - y * s_a
        pts[:,1] = x * s_a + y * c_a
        self.north_arrow.setData(pos=pts)
        
        tip = np.array([0, 6, 0.5])
        tx = tip[0] * c_a - tip[1] * s_a
        ty = tip[0] * s_a + tip[1] * c_a
        self.txt_north.setData(pos=[tx, ty, 0.5])
        
        # Geometry Effective Dims
        eff_w = s.geometry.width
        eff_l = s.geometry.length
        eff_t = s.geometry.thickness
        if self.show_tolerance:
             tol = s.config.tolerance
             eff_w += tol
             eff_l += tol
             eff_t += tol
             
        # Generate Effective Box
        current_verts, current_faces = self._create_box_verts(eff_w, eff_l, eff_t)
        
        # Generate Edge verts (Thick Purple Line at Top)
        # User requested "other side" -> -eff_l/2 
        # "Top glass edge" -> z = +eff_t/2
        edge_th = 0.05 # 5cm thickness
        edge_verts, edge_faces = self._create_box_verts(eff_w, edge_th, edge_th)
        
        # Shift to Top Edge of Glass (-Y edge, +Z face)
        edge_shift = np.array([0, -eff_l/2.0, eff_t/2.0]) 
        edge_verts = edge_verts + edge_shift
        
        # Culling / View Mode Logic
        if not self.show_full_plant:
             # Representative 3x3 View
             r_start, r_end = 0, 3
             c_start, c_end = 0, 3
             view_h, view_w = 3, 3
             
             # Center the 3x3 grid
             off_x = (view_w - 1) * pitch_x / 2
             off_y = (view_h - 1) * pitch_y / 2
        else:
             # Full Plant View
             r_start, r_end = 0, rows
             c_start, c_end = 0, cols
             
             # Calculate total width/height with gaps
             f_sp_x = getattr(s.config, 'field_spacing_x', pitch_x)
             f_sp_y = getattr(s.config, 'field_spacing_y', pitch_y)
             
             last_col = cols - 1
             l_field_c = last_col // 4
             l_local_c = last_col % 4
             field_stride_x = (3 * pitch_x) + f_sp_x
             total_w = (l_field_c * field_stride_x) + (l_local_c * pitch_x)
             off_x = total_w / 2.0

             last_row = rows - 1
             l_field_r = last_row // 4
             l_local_r = last_row % 4
             field_stride_y = (3 * pitch_y) + f_sp_y
             total_h = (l_field_r * field_stride_y) + (l_local_r * pitch_y)
             off_y = total_h / 2.0
        
        # Buffers
        all_verts = []
        all_faces = []
        all_colors = []
        pivot_verts = []
        pivot_faces = []
        sh_verts = []
        sh_faces = []
        cyl_verts_list = []
        cyl_faces_list = []
        ray_pts_sun_list = []
        edge_verts_list = []
        edge_faces_list = []
        
        v_idx = 0
        e_idx = 0
        p_idx = 0
        sh_idx = 0
        cyl_idx = 0
        
        # Viz Constants
        c_pastel_blue = (0.4, 0.7, 1.0, 1) 
        c_edge = (0.8, 0.8, 0.8, 1)
        c_stow = (1.0, 0.6, 0.0, 1)
        c_clash = (1.0, 0.0, 0.0, 1)
        c_purple = (0.6, 0.0, 0.8, 1)
        
        offset_vec = np.array(s.geometry.pivot_offset)
        pv = self.ico_verts
        pf = self.ico_faces
        z_face = eff_t/2.0 + 0.02 
        
        # Actual dims for ray spacing
        w_act = s.geometry.width
        l_act = s.geometry.length
        ray_grid = []
        xm = w_act/2 * 0.75
        ym = l_act/2 * 0.75
        for rx in [-xm, 0, xm]:
             for ry in [-ym, 0, ym]:
                  ray_grid.append(np.array([rx, ry, 0.0]))
                  
        cv_unit, cf_unit = self.cyl_verts, self.cyl_faces
        
        # Local Tracking Logic
        local_sun_az = eff_sun_az - (-s.config.plant_rotation)
        R_track = self._get_rotation_matrix(local_sun_az, 90.0 - eff_sun_el)
        
        is_kernel_mode = (states is not None and len(states) == 9)
        use_checkerboard = False
        if is_kernel_mode:
             center_st = states[4]
             if center_st.mode and ("STOW" in center_st.mode):
                  use_checkerboard = True
        
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                
                # Logic: State Mapping
                if is_kernel_mode:
                    if not self.show_full_plant:
                        # Representative View Mapping (0..2)
                        is_top = (r == 0)
                        is_bot = (r == 2)
                        is_left = (c == 0)
                        is_right = (c == 2)
                        parity = (r + c) % 2
                    else:
                        # Full Plant Mapping
                        is_top = (r == 0)
                        is_bot = (r == rows - 1)
                        is_left = (c == 0)
                        is_right = (c == cols - 1)
                        parity = (r + c) % 2 
                    
                    kr, kc = 1, 1 
                    if is_top:
                        kr = 0
                        if is_left: kc = 0
                        elif is_right: kc = 2
                        else:
                            if not is_left and not is_right:
                                  kc = 1 if (use_checkerboard and parity==1) else (0 if use_checkerboard else 1)
                    elif is_bot:
                        kr = 2
                        if is_left: kc = 0
                        elif is_right: kc = 2
                        else: kc = 1 if (use_checkerboard and parity==1) else (0 if use_checkerboard else 1)
                    elif is_left:
                        kc = 0
                        kr = 1 if (use_checkerboard and parity==1) else (0 if use_checkerboard else 1)
                    elif is_right:
                        kc = 2
                        kr = 1 if (use_checkerboard and parity==1) else (0 if use_checkerboard else 1)
                    else:
                        if use_checkerboard: kr, kc = (1, 1) if parity == 0 else (1, 2)
                        else: kr, kc = 1, 1
                    idx_k = kr * 3 + kc
                    st = states[idx_k]
                    
                else: 
                     # Non-Kernel Mode (Full states array)
                     idx_glb = r * cols + c
                     if states and idx_glb < len(states):
                         st = states[idx_glb]
                     else:
                         st = None
                    
                st = st if 'st' in locals() else None 
                
                # Defaults
                R = R_track
                color = c_pastel_blue
                shad_polys = []
                
                if st:
                     R = st.rotation
                     if st.collision: color = c_clash
                     if st.mode and ("STOW" in st.mode): color = c_stow
                     if st.shadow_polys: shad_polys = st.shadow_polys
                
                # FIELD SPACING LOGIC
                # Fields are 4x4.
                # Pitch is uniform within a field.
                # But between Field Col 3 and Field Col 4 (Local 3->0), we add field_spacing_x.
                # Actually field_spacing is "Pivot to Pivot" between fields.
                # Standard spacing is pitch.
                # So the "jump" is field_spacing.
                # Coordinate = (field_idx * field_stride) + (local_idx * pitch)
                # Field Stride = (3 * pitch) + field_spacing
                
                # However, if field_spacing isn't defined, default to pitch? 
                # Config has default.
                
                f_sp_x = getattr(s.config, 'field_spacing_x', pitch_x)
                f_sp_y = getattr(s.config, 'field_spacing_y', pitch_y)
                
                # X Calc
                field_c = c // 4
                local_c = c % 4
                field_stride_x = (3 * pitch_x) + f_sp_x
                px = (field_c * field_stride_x) + (local_c * pitch_x) - off_x
                
                # Y Calc
                field_r = r // 4
                local_r = r % 4
                field_stride_y = (3 * pitch_y) + f_sp_y
                py = (field_r * field_stride_y) + (local_r * pitch_y) - off_y
                
                pz = s.geometry.pivot_offset[2] + 2.0
                pos = np.array([px, py, pz])
                
                # 1. Panel
                v_shifted = current_verts + offset_vec
                v_rot = v_shifted @ R.T
                v_final = v_rot + pos
                all_verts.append(v_final)
                all_faces.append(current_faces + v_idx)
                for k in range(12):
                     all_colors.append(color if k in [2,3] else c_edge)
                v_idx += 8
                
                # 1b. Edge (Purple)
                v_ed_shifted = edge_verts + offset_vec
                v_ed_rot = v_ed_shifted @ R.T
                v_ed_final = v_ed_rot + pos
                edge_verts_list.append(v_ed_final)
                edge_faces_list.append(edge_faces + e_idx)
                e_idx += 8
                
                # 2. Pivot
                if self.show_pivots:
                     pivot_verts.append(pv + pos)
                     pivot_faces.append(pf + p_idx)
                     p_idx += len(pv)
                
                # 3. Shadows (Prismatic)
                if shad_polys:
                     norm = R[:,2]
                     
                     shift_pos = pos - st.position
                     rot_offset = offset_vec @ R.T
                     
                     dist_from_pivot_plane = np.dot(rot_offset, norm) + s.geometry.thickness/2.0 + 0.02
                     total_lift = norm * dist_from_pivot_plane
                     thickness_vec = norm * 0.005 # 5mm
                     
                     for poly in shad_polys:
                          if len(poly)<3: continue
                          pp = np.array(poly)
                          if pp.shape[1]==2: pp=np.column_stack((pp, np.zeros(len(pp))))
                          
                          # 1. To Plant Frame
                          if is_kernel_mode: pp = pp + shift_pos
                          
                          # 2. Lift to Surface (Bottom of prism)
                          pp_bot = pp + total_lift
                          pp_top = pp_bot + thickness_vec
                          
                          # Top Face
                          nv = len(pp_top)
                          for k in range(1, nv-1): 
                               sh_verts.extend([pp_top[0], pp_top[k], pp_top[k+1]])
                               sh_faces.append([sh_idx, sh_idx+1, sh_idx+2])
                               sh_idx += 3
                               
                          # Side Faces
                          for k in range(nv):
                               next_k = (k+1)%nv
                               p1, p2 = pp_top[k], pp_top[next_k]
                               p3, p4 = pp_bot[next_k], pp_bot[k]
                               
                               sh_verts.extend([p1, p2, p3])
                               sh_faces.append([sh_idx, sh_idx+1, sh_idx+2])
                               sh_idx += 3
                               
                               sh_verts.extend([p1, p3, p4])
                               sh_faces.append([sh_idx, sh_idx+1, sh_idx+2])
                               sh_idx += 3

                # 4. RAYS
                if self.show_rays:
                     off_g = offset_vec @ R.T
                     pos_center = pos + off_g
                     
                     norm = R[:, 2] 
                     norm_neg = -norm 
                     
                     src = np.array([0,0,1])
                     axis = np.cross(src, norm_neg)
                     dot = np.dot(src, norm_neg)
                     R_cyl = np.eye(3)
                     if np.linalg.norm(axis) > 1e-6:
                          axis = axis / np.linalg.norm(axis)
                          angle = np.arccos(np.clip(dot, -1, 1))
                          K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                          R_cyl = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
                     elif dot < 0:
                          R_cyl = np.diag([1,-1,-1])
                     
                     for rg in ray_grid: # 9 points
                          rg_r = rg @ R.T
                          
                          p_top = pos_center + rg_r + norm * z_face
                          p_end = p_top + norm * 5.0
                          ray_pts_sun_list.append(p_top)
                          ray_pts_sun_list.append(p_end)
                          
                          p_bot = pos_center + rg_r - norm * z_face
                          
                          if abs(norm_neg[2]) > 0.001: 
                              t = -p_bot[2] / norm_neg[2]
                              if t > 0:
                                   cv_scaled = cv_unit * np.array([1,1,t])
                                   cv_rot = cv_scaled @ R_cyl.T
                                   cv_final = cv_rot + p_bot
                                   cyl_verts_list.append(cv_final)
                                   cyl_faces_list.append(cf_unit + cyl_idx)
                                   cyl_idx += len(cv_unit)

        # --- DRAW ---
        if all_verts:
            self.plant_mesh.setMeshData(vertexes=np.vstack(all_verts), faces=np.vstack(all_faces), faceColors=np.array(all_colors))
        else:
            self.plant_mesh.setMeshData(vertexes=[], faces=[])
            
        if self.show_pivots and pivot_verts:
            self.pivot_mesh.setMeshData(vertexes=np.vstack(pivot_verts), faces=np.vstack(pivot_faces))
            self.pivot_mesh.setVisible(True)
        else:
            self.pivot_mesh.setVisible(False)
             
        # Update Edge Mesh
        if edge_verts_list:
            ev = np.vstack(edge_verts_list)
            ef = np.vstack(edge_faces_list)
            md_edge = gl.MeshData(vertexes=ev, faces=ef)
            self.edge_mesh.setMeshData(meshdata=md_edge)
            self.edge_mesh.setVisible(True)
        else:
            self.edge_mesh.setVisible(False)
            
        if sh_verts:
            self.panel_shadow_mesh.setMeshData(vertexes=np.array(sh_verts), faces=np.array(sh_faces))
            self.panel_shadow_mesh.setVisible(True)
        else:
            self.panel_shadow_mesh.setVisible(False)
            
        if self.show_rays and ray_pts_sun_list:
             self.rays_sun.setData(pos=np.vstack(ray_pts_sun_list))
             self.rays_sun.setVisible(True)
             
             if cyl_verts_list:
                  self.rays_gnd_mesh.setMeshData(vertexes=np.vstack(cyl_verts_list), faces=np.vstack(cyl_faces_list))
                  self.rays_gnd_mesh.setVisible(True)
        else:
             self.rays_sun.setVisible(False)
             self.rays_gnd_mesh.setVisible(False)

    def _get_rotation_matrix(self, az_deg, el_deg):
        raz = np.radians(-az_deg)
        cz, sz = np.cos(raz), np.sin(raz)
        Rz = np.array([[cz, -sz, 0], [sz,  cz, 0], [ 0,   0, 1]])
        rel = np.radians(el_deg)
        cx, sx = np.cos(rel), np.sin(rel)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        return Rz @ Rx
