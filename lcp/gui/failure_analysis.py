
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QSlider, QGroupBox, QPushButton, QFrame, QScrollArea)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

import numpy as np
import math
from datetime import datetime

from lcp.gui.state import AppState
from lcp.physics.engine import InfiniteKernel, PanelState

class FailureAnalysisWidget(QWidget):
    """
    Widget for visualizing Mechanical Failure in a 4x4 field.
    """
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.viewport = None # Injected
        
        # Internal Data
        self.day_frames = [] # List of frames for selected day
        self.roi_kernel = None # For local collision calc
        
        # Grid Params
        self.field_w = 4
        self.field_h = 4
        
        self.setup_ui()
        
    def set_viewport(self, viewport):
        self.viewport = viewport
        
    def load_data(self, day_frames):
        self.day_frames = [f for f in day_frames if f['sun_el'] > 0] # Daylight only
        if not self.day_frames: return
        
        # Setup Sliders
        self.sl_global.blockSignals(True)
        self.sl_global.setRange(0, len(self.day_frames)-1)
        self.sl_global.setValue(0)
        self.sl_global.blockSignals(False)
        
        self.sl_fail.blockSignals(True)
        self.sl_fail.setRange(0, len(self.day_frames)-1)
        self.sl_fail.setValue(0) 
        self.sl_fail.blockSignals(False)
        
        # Populate Fields
        self.refresh_fields()
        
        # Update
        self.update_viz()
        
    def refresh_fields(self):
        rows = self.state.rows
        cols = self.state.cols
        
        n_fr = math.ceil(rows / self.field_h)
        n_fc = math.ceil(cols / self.field_w)
        total = n_fr * n_fc
        
        self.cb_field.blockSignals(True)
        self.cb_field.clear()
        for i in range(1, total + 1):
             self.cb_field.addItem(f"Field {i}")
        self.cb_field.blockSignals(False)
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Controls
        gb = QGroupBox("Failure Scenarios")
        gb_l = QVBoxLayout()
        
        # 1. Field Selector
        h_field = QHBoxLayout()
        h_field.addWidget(QLabel("Failure Field (4x4):"))
        self.cb_field = QComboBox()
        self.cb_field.currentIndexChanged.connect(self.update_viz)
        h_field.addWidget(self.cb_field)
        gb_l.addLayout(h_field)
        
        # 2. Global Time
        gb_l.addWidget(QLabel("Plant Operation (Global Time):"))
        self.lbl_global_t = QLabel("--:--")
        self.lbl_global_t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_global_t.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        gb_l.addWidget(self.lbl_global_t)
        
        self.sl_global = QSlider(Qt.Orientation.Horizontal)
        self.sl_global.valueChanged.connect(self.update_viz)
        gb_l.addWidget(self.sl_global)
        
        # 3. Fail Time
        gb_l.addWidget(QLabel("Failed Field Position (Stuck At):"))
        self.lbl_fail_t = QLabel("--:--")
        self.lbl_fail_t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_fail_t.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.lbl_fail_t.setStyleSheet("color: red")
        gb_l.addWidget(self.lbl_fail_t)
        
        self.sl_fail = QSlider(Qt.Orientation.Horizontal)
        self.sl_fail.setStyleSheet("""
            QSlider::handle:horizontal {
                background: red;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0; 
                border-radius: 3px;
            }
        """)
        self.sl_fail.valueChanged.connect(self.update_viz)
        gb_l.addWidget(self.sl_fail)
        
        # 4. Info
        self.lbl_info = QLabel("Select a field to simulate failure.\nThe rest of the plant tracks normally.")
        self.lbl_info.setWordWrap(True)
        gb_l.addWidget(self.lbl_info)
        
        gb.setLayout(gb_l)
        layout.addWidget(gb)
        
        layout.addStretch()
        
    def update_viz(self):
        if not self.day_frames: 
            print("FailureAnalysisWidget: No day frames.")
            return
        if not self.viewport:
            print("FailureAnalysisWidget: No viewport.")
            return
        
        # Get Indices
        g_idx = self.sl_global.value()
        f_idx = self.sl_fail.value()
        
        g_frame = self.day_frames[g_idx]
        f_frame = self.day_frames[f_idx]
        
        # Update Labels
        self.lbl_global_t.setText(g_frame['time'].strftime("%H:%M"))
        self.lbl_fail_t.setText(f_frame['time'].strftime("%H:%M"))
        
        # Field Calc
        field_id = self.cb_field.currentIndex() # 0-indexed
        
        rows = self.state.rows
        cols = self.state.cols
        n_fc = math.ceil(cols / self.field_w)
        
        fr = field_id // n_fc
        fc = field_id % n_fc
        
        r_start = fr * self.field_h
        r_end = min((fr + 1) * self.field_h, rows)
        c_start = fc * self.field_w
        c_end = min((fc + 1) * self.field_w, cols)
        
        # Define ROI (Include Boundary)
        roi_r_start = max(0, r_start - 1)
        roi_r_end = min(rows, r_end + 1)
        roi_c_start = max(0, c_start - 1)
        roi_c_end = min(cols, c_end + 1)
        
        # Params
        rot = -self.state.config.plant_rotation
        
        # Global (Healthy)
        g_sun_az = g_frame['sun_az']
        g_sun_el = g_frame['sun_el']
        g_local_az = g_sun_az - rot
        
        # Failed (Stuck)
        f_sun_az = f_frame['sun_az'] # Sun pos at fail time (determines tracking angle)
        f_sun_el = f_frame['sun_el']
        f_local_az = f_sun_az - rot
        
        # We need to simulate the ROI panels.
        # Kernel normally solves 3x3. Here we need arbitrary N x M.
        # Since we just want visual clash detection, we can use the Kernel's logic manually or loop.
        
        # Lazy Init ROI Kernel
        if not self.roi_kernel:
             self.roi_kernel = InfiniteKernel(self.state.geometry, self.state.config)
             # Pre-calc pivots?
             # InfiniteKernel assumes 3x3 relative.
             # We can use it per panel by constructing 3x3 neighborhoods on the fly?
             # Or just use `collider` directly if exposed?
             # InfiniteKernel.solve_timestep does 3x3.
             pass
             
                  
        # 1. Calculate Rotations
        from lcp.physics.kinematics import AzElRig
        rig = AzElRig()
        
        # Healthy Rot
        R_healthy = rig.get_orientation(g_local_az, g_sun_el)
        # Failed Rot
        R_failed = rig.get_orientation(f_local_az, f_sun_el)

        # Construct States List (Sparse)
        # Size: rows * cols
        # To ensure visual consistency, we use a shared "Healthy" state for background panels
        # This forces them to use AzElRig rotation instead of Viewport's default
        healthy_state = PanelState(
            index=(-1,-1),
            position=np.zeros(3),
            rotation=R_healthy,
            mode="TRACKING",
            collision=False,
            theta_loss=0.0,
            stow_loss=0.0,
            shadow_loss=0.0,
            shadow_polys=[],
            power_factor=1.0
        )
        states = [healthy_state] * (rows * cols)
        
        # We need to compute clashes.
        # Efficient way:
        # Iterate ROI panels.
        # For each panel, determine if it is FAILED or HEALTHY.
        # Calculate its rotation R.
        # Then check collision with neighbors.
        
        roi_rotations = {} # (r, c) -> R_matrix
        
        for r in range(roi_r_start, roi_r_end):
             for c in range(roi_c_start, roi_c_end):
                  # Is it in the failed field?
                  is_failed = (r_start <= r < r_end) and (c_start <= c < c_end)
                  roi_rotations[(r,c)] = R_failed if is_failed else R_healthy
                  
        # 2. Check Collisions
        # We check each panel in ROI against its neighbors in ROI.
        # If neighbor is outside ROI, we assume it is Healthy (R_healthy).
        
        # We can reuse `Collider` logic but we need relative pivots.
        # Let's verify each panel in ROI
        
        # Precompute pivot offsets
        px = self.state.config.grid_pitch_x
        py = self.state.config.grid_pitch_y
        
        for r in range(roi_r_start, roi_r_end):
             for c in range(roi_c_start, roi_c_end):
                  
                  # Create State
                  st = PanelState(
                      index=(r,c),
                      position=np.zeros(3), # Not used for basic viz
                      rotation=roi_rotations[(r,c)],
                      mode="TRACKING",
                      collision=False,
                      theta_loss=0.0,
                      stow_loss=0.0,
                      shadow_loss=0.0,
                      shadow_polys=[],
                      power_factor=0.0
                  )
                  
                  # Check vs Neighbors
                  # Neighbors: (r, c+1), (r+1, c), (r+1, c+1), (r+1, c-1) ...
                  # We only need to check unique pairs, but here we construct state per panel.
                  # A panel clashes if ANY neighbor clashes with it.
                  
                  # Define neighbors to check (8-way or just cardinal?)
                  # Tracker collisions usually N-S (End-to-End) or E-W (Shading/Edge).
                  # Let's check all 8 to be safe or just use Kernel's standard set?
                  # Kernel checks: N, S, E, W, NE, NW, SE, SW.
                  
                  deltas = [
                      (-1, 0), (1, 0), (0, -1), (0, 1), # Cardinal
                      (-1, -1), (-1, 1), (1, -1), (1, 1) # Diagonal
                  ]
                  
                  my_R = st.rotation
                  
                  for dr, dc in deltas:
                       nr, nc = r + dr, c + dc
                       
                       # Neighbor Rotation
                       if (nr, nc) in roi_rotations:
                            n_R = roi_rotations[(nr, nc)]
                       else:
                            # Outside ROI -> Healthy
                            n_R = R_healthy 
                            
                       # Relative Pivot
                       # Need to account for Field Spacing!
                       # We calculate absolute positions for r,c and nr,nc (with offset 0) and subtract.
                       
                       def get_abs_pos(row, col):
                            f_sp_x = getattr(self.state.config, 'field_spacing_x', px)
                            f_sp_y = getattr(self.state.config, 'field_spacing_y', py)
                            
                            fc = col // 4
                            lc = col % 4
                            stride_x = (3 * px) + f_sp_x
                            x = (fc * stride_x) + (lc * px)
                            
                            fr = row // 4
                            lr = row % 4
                            stride_y = (3 * py) + f_sp_y
                            y = (fr * stride_y) + (lr * py)
                            return np.array([x, y, 0.0])

                       pos_me = get_abs_pos(r, c)
                       pos_neighbor = get_abs_pos(nr, nc)
                       rel_pos = pos_neighbor - pos_me
                       
                       # Call Collider
                       # We use local frame: Pivot A = (0,0,0), Pivot B = rel_pos
                       pivot_a = np.zeros(3)
                       if self.roi_kernel.collider.check_clash(pivot_a, rel_pos, my_R, rot_b=n_R):
                            st.collision = True
                            break
                            
                  # Store State
                  idx = r * cols + c
                  states[idx] = st
                  
        # 3. Update Viewport
        self.viewport.set_show_full_plant(True)
        # "Safety" flag for viewport determines border color mostly, let's say False if any collision
        any_clash = any(s.collision for s in states if s)
        
        self.viewport.update_from_frame(g_sun_az, g_sun_el, not any_clash, states)
        
        # Update Info
        self.lbl_info.setText(f"Simulating Field {field_id + 1}\n(Rows {r_start}-{r_end-1}, Cols {c_start}-{c_end-1})\n\n"
                              f"Clashes Detected: {'YES' if any_clash else 'NO'}")
                              
