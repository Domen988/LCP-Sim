
import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtCore import pyqtSignal, QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from lcp.gui.state import AppState

class PlantViewport(gl.GLViewWidget):
    """
    The 3D OpenGL Viewport. 
    Renders the plant using high-performance GLMeshItem.
    """
    
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        
        # Camera Setup
        self.setCameraPosition(distance=40, elevation=30, azimuth=-90)
        self.setBackgroundColor('w') # White Background
        
        # Grid
        g = gl.GLGridItem()
        g.setSize(x=100, y=100)
        g.setSpacing(x=5, y=5)
        g.setColor((200, 200, 200, 255)) # Light Grey for contrast on white
        self.addItem(g)
        
        # Axis
        # ax = gl.GLAxisItem()
        # self.addItem(ax)
        
        # The Plant Mesh
        self.plant_mesh = gl.GLMeshItem(
            meshdata=None, 
            smooth=False, 
            shader='shaded', 
            color=(0.1, 0.4, 0.7, 1), # Blue panels
            glOptions='opaque'
        )
        self.addItem(self.plant_mesh)
        
        # Precompute base geometry
        self.base_verts, self.base_faces = self._create_unit_panel()
        
        # Initial Draw
        self.update_scene()
        
    def update_from_frame(self, sun_az, sun_el):
        """
        Fast update from simulation frame replay.
        Does NOT update AppState, just Visuals.
        """
        # We temporarily override the state for rendering
        # Or better, pass explicit args to update_scene
        self.update_scene(explicit_az=sun_az, explicit_el=sun_el)
        
    def _create_unit_panel(self):
        w = self.state.geometry.width
        l = self.state.geometry.length
        # 4 corners centered
        verts = np.array([
            [-w/2, -l/2, 0],
            [ w/2, -l/2, 0],
            [ w/2,  l/2, 0],
            [-w/2,  l/2, 0]
        ])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        return verts, faces
        
    def update_scene(self, explicit_az=None, explicit_el=None):
        """Re-calculates vertex positions based on current state or explicit override."""
        
        s = self.state
        
        if explicit_az is not None:
             eff_sun_az = explicit_az
             eff_sun_el = explicit_el
        else:
             eff_sun_az = s.sun_az
             eff_sun_el = s.sun_el
             
        # Plant Rotation
        local_sun_az = eff_sun_az - s.plant_rotation
        local_stow_az = s.stow_az - s.plant_rotation
        
        rows, cols = s.rows, s.cols
        pitch_x = s.config.grid_pitch_x
        pitch_y = s.config.grid_pitch_y
        
        # Offsets to center the grid
        off_x = (cols - 1) * pitch_x / 2
        off_y = (rows - 1) * pitch_y / 2
        
        # Pre-calculate Rotation Matrices
        # We only have 2 unique orientations per frame usually (Odd vs Even)
        # 1. Stow Orientation
        R_stow = self._get_rotation_matrix(local_stow_az, s.stow_el)
        # 2. Track Orientation (Simple tracking for Viz)
        # We want Normal pointing to Sun. Sun Elevation 90 (Zenith) -> Tilt 0 (Flat).
        # Sun Elevation 0 (Horizon) -> Tilt 90 (Vertical).
        R_track = self._get_rotation_matrix(local_sun_az, 90.0 - eff_sun_el)
        
        all_verts = []
        all_faces = []
        idx = 0
        
        # Vectorized approach or Loop? Loop is easier to read first.
        # Ideally we use Instanced Rendering, but pyqtgraph GLMeshItem implies single mesh.
        # We can construct the big mesh.
        
        for r in range(rows):
            for c in range(cols):
                is_stow_row = ((r + c) % 2 == 0)
                
                # Choose Matrix
                R = R_stow if is_stow_row else R_track
                
                # Transform Verts
                # (4,3) . (3,3)
                t_verts = self.base_verts @ R.T
                
                # Translate
                px = (c * pitch_x) - off_x
                py = (r * pitch_y) - off_y
                pz = s.geometry.pivot_offset[2] + 2.0
                
                t_verts += np.array([px, py, pz])
                
                all_verts.append(t_verts)
                # Offset faces by vertex count (4 per panel)
                all_faces.append(self.base_faces + (idx * 4))
                idx += 1
                
        # Merge
        if all_verts:
            start_v = np.vstack(all_verts)
            start_f = np.vstack(all_faces)
            self.plant_mesh.setMeshData(vertexes=start_v, faces=start_f)
            
    def _get_rotation_matrix(self, az_deg, el_deg):
        # Azimuth (Z-axis, negative for CW)
        raz = np.radians(-az_deg)
        cz, sz = np.cos(raz), np.sin(raz)
        Rz = np.array([
            [cz, -sz, 0],
            [sz,  cz, 0],
            [ 0,   0, 1]
        ])
        
        # Elevation (X-axis, tilt)
        rel = np.radians(el_deg)
        cx, sx = np.cos(rel), np.sin(rel)
        Rx = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ])
        
        return Rz @ Rx
