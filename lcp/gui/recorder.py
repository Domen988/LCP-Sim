
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
                             QPushButton, QGroupBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QSplitter, QLineEdit, QComboBox, QMessageBox, 
                             QAbstractItemView, QFrame, QSizePolicy, QFileDialog, QApplication)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

from lcp.gui.state import AppState
from lcp.core.stow import StowProfile
from lcp.physics.engine import InfiniteKernel

class StowRecorder(QWidget):
    """
    New Stow Recorder Tab.
    Right: Table (Time, Safety, SunAz, SunEl, StowAz, StowEl, ProfileSafety)
    Left: Controls (Header, Slider, Teach Pendant, Actions, File IO)
    """
    
    # Signals
    # Emits when manual stow position changes (for 3D viz)
    # args: sun_az, sun_el, safety_status (bool), states_list (from kernel)
    preview_update = pyqtSignal(float, float, bool, list)
    
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        
        # Internal Data
        self.day_data = None # DataFrame for the selected day
        self.current_idx = 0
        self.kernel = None # Will be injected
        self.is_playing = False
        
        # Playback Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_forward)
        self.timer.setInterval(100) # 10 FPS
        
        self.setup_ui()

        # Set default profile folder and refresh
        app_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_profile_folder = os.path.join(app_root, "saved_stow_profiles")
        if not os.path.exists(default_profile_folder):
            try:
                os.makedirs(default_profile_folder)
            except OSError:
                default_profile_folder = os.getcwd() # Fallback if creation fails
        self.txt_path.setText(default_profile_folder)
        self.refresh_profiles()
        
    def set_kernel(self, kernel: InfiniteKernel):
        self.kernel = kernel
        
    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # --- LEFT: CONTROLS ---
        left_widget = QWidget()
        l_layout = QVBoxLayout(left_widget)
        
        # 1. Header (Day Info)
        gb_info = QGroupBox("Context")
        info_l = QVBoxLayout()
        self.lbl_date = QLabel("Date: --")
        self.lbl_geo = QLabel("Geo: --") # Will contain dims
        self.lbl_pitch = QLabel("Grid: --")
        self.lbl_tol = QLabel("Tol: --")
        info_l.addWidget(self.lbl_date)
        info_l.addWidget(self.lbl_geo)
        info_l.addWidget(self.lbl_pitch)
        info_l.addWidget(self.lbl_tol)
        gb_info.setLayout(info_l)
        l_layout.addWidget(gb_info)
        
        # 2. Timeline Control
        gb_time = QGroupBox("Timeline")
        time_l = QVBoxLayout()
        
        self.lbl_time = QLabel("--:--")
        self.lbl_time.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.sl_time = QSlider(Qt.Orientation.Horizontal)
        self.sl_time.setRange(0, 100)
        self.sl_time.valueChanged.connect(self.on_time_slider)
        
        h_play = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play Day")
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        h_play.addWidget(self.btn_play)
        h_play.addWidget(self.btn_pause)
        
        time_l.addWidget(self.lbl_time)
        time_l.addWidget(self.sl_time)
        time_l.addLayout(h_play)
        gb_time.setLayout(time_l)
        l_layout.addWidget(gb_time)
        
        # 3. Teach Pendant (Stow Sliders)
        gb_teach = QGroupBox("Teach Position")
        teach_l = QVBoxLayout()
        
        # Azimuth
        self.lbl_az = QLabel("Stow Az: 0.0°")
        self.sl_az = QSlider(Qt.Orientation.Horizontal)
        self.sl_az.setRange(0, 3600) # 0 to 360
        self.sl_az.valueChanged.connect(self.on_teach_change)
        
        # Elevation
        self.lbl_el = QLabel("Stow El: 0.0°")
        self.sl_el = QSlider(Qt.Orientation.Horizontal)
        self.sl_el.setRange(0, 900)
        self.sl_el.valueChanged.connect(self.on_teach_change)
        
        teach_l.addWidget(self.lbl_az)
        teach_l.addWidget(self.sl_az)
        teach_l.addWidget(self.lbl_el)
        teach_l.addWidget(self.sl_el)
        gb_teach.setLayout(teach_l)
        l_layout.addWidget(gb_teach)
        
        # 4. Actions
        gb_act = QGroupBox("Actions")
        act_l = QVBoxLayout()
        
        self.btn_copy_next = QPushButton("Copy & Next Row")
        self.btn_copy_next.clicked.connect(self.copy_and_next)
        
        h_interp = QHBoxLayout()
        self.btn_interp = QPushButton("Interpolate Selection")
        self.btn_interp.clicked.connect(self.interpolate_selection)
        self.btn_reset = QPushButton("Reset Row (Sun)")
        self.btn_reset.clicked.connect(self.reset_row)
        h_interp.addWidget(self.btn_interp)
        h_interp.addWidget(self.btn_reset)
        
        act_l.addWidget(self.btn_copy_next)
        act_l.addLayout(h_interp)
        
        # Smooth Playback Controls
        self.btn_play_smooth = QPushButton("▶ Play Selected (Smooth)")
        self.btn_play_smooth.clicked.connect(self.play_smooth)
        act_l.addWidget(self.btn_play_smooth)
        
        self.lbl_smooth = QLabel("Smooth Scrub:")
        self.sl_smooth = QSlider(Qt.Orientation.Horizontal)
        self.sl_smooth.setRange(0, 100)
        self.sl_smooth.setEnabled(False)
        self.sl_smooth.valueChanged.connect(self.on_smooth_slider)
        
        act_l.addWidget(self.lbl_smooth)
        act_l.addWidget(self.sl_smooth)
        
        gb_act.setLayout(act_l)
        l_layout.addWidget(gb_act)
        
        # 5. Profile Management
        gb_file = QGroupBox("Stow Profile")
        file_l = QVBoxLayout()
        
        h_path = QHBoxLayout()
        self.txt_path = QLineEdit(os.path.join(os.getcwd(), "profiles"))
        btn_browse = QPushButton("...")
        btn_browse.clicked.connect(self.browse_folder)
        h_path.addWidget(self.txt_path)
        h_path.addWidget(btn_browse)
        
        h_name = QHBoxLayout()
        self.txt_name = QLineEdit("MyStowProfile")
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self.save_profile)
        h_name.addWidget(self.txt_name)
        h_name.addWidget(self.btn_save)
        
        self.btn_reset_all = QPushButton("Reset Entire Profile")
        self.btn_reset_all.clicked.connect(self.reset_all_rows)
        file_l.addWidget(self.btn_reset_all)
        
        h_load = QHBoxLayout()
        self.cb_profiles = QComboBox()
        self.btn_load = QPushButton("Load")
        self.btn_load.clicked.connect(self.load_profile)
        self.btn_del = QPushButton("Delete")
        self.btn_del.setStyleSheet("color: red")
        self.btn_del.clicked.connect(self.delete_profile)
        h_load.addWidget(self.cb_profiles)
        h_load.addWidget(self.btn_load)
        h_load.addWidget(self.btn_del)
        
        # Populate Profiles (will be called in __init__ after txt_path is set)
        # self.refresh_profiles() 
        
        file_l.addLayout(h_path)
        file_l.addLayout(h_name)
        file_l.addLayout(h_load)
        gb_file.setLayout(file_l)
        l_layout.addWidget(gb_file)
        
        l_layout.addStretch()
        splitter.addWidget(left_widget)
        
        # --- RIGHT: TABLE ---
        self.table = QTableWidget()
        cols = ["Time", "Sim Safety", "Sun Az", "Sun El", "Stow Az", "Stow El", "Profile Safety"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.cellChanged.connect(self.on_table_edit)
        self.table.selectionModel().selectionChanged.connect(self.on_table_select)
        
        splitter.addWidget(self.table)
        splitter.setSizes([400, 800])
        splitter.setStretchFactor(1, 2)
        
    def load_day_data(self, day_df, date_label):
        """
        Populate Table from Sim Results.
        """
        self.day_data = day_df.copy().reset_index(drop=True)
        self.lbl_date.setText(f"Date: {date_label}")
        geo = self.state.geometry
        cfg = self.state.config
        
        # Pivot Z Correction (Offset + Half Thickness)
        piv_z = geo.pivot_offset[2] + (geo.thickness / 2.0)
        
        self.lbl_geo.setText(f"Panel: {geo.width}x{geo.length}m, Thk: {geo.thickness}m")
        self.lbl_pitch.setText(f"Pitch: {cfg.grid_pitch_x}x{cfg.grid_pitch_y}m, Pivot Z: {piv_z:.3f}m")
        self.lbl_tol.setText(f"Tolerance: {cfg.tolerance}m")
        
        # Configure Sliders
        rows = len(self.day_data)
        self.sl_time.blockSignals(True)
        self.sl_time.setRange(0, rows - 1)
        self.sl_time.setValue(0)
        self.sl_time.blockSignals(False)
        
        # Populate Table
        self.table.blockSignals(True)
        self.table.setRowCount(rows)
        
        for r in range(rows):
            row = self.day_data.iloc[r]
            
            # We want: Safe=Check, Clash=X
            # If row['safety'] (Collision) is True -> Clash -> X
            # Note: During Load, we might have 'sim_safety' string already.
            # Handled below.
            
            # 0: Time
            if hasattr(row['time'], 'strftime'):
                 time_str = row['time'].strftime("%H:%M")
            else:
                 time_str = str(row['time'])
                 
            item_t = QTableWidgetItem(time_str)
            item_t.setFlags(item_t.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 0, item_t)
            
            # 1: Sim Safety
            # In DataFrame we expect 'safety' col (bool) OR 'sim_safety' (str/emoji from file)
            # If loading from Simulation Results: 'safety' is bool (True=Clash).
            # If loading from File: 'sim_safety' is "✅"/"❌".
            if 'sim_safety' in row:
                safety = row['sim_safety']
            elif 'safety' in row:
                 safety = "❌" if row['safety'] else "✅"
            else:
                 safety = "❓"
                 
            item_s = QTableWidgetItem(safety)
            item_s.setFlags(item_s.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 1, item_s)
            
            # 2/3: Sun Az/El
            item_saz = QTableWidgetItem(f"{row['sun_az']:.1f}")
            item_saz.setFlags(item_saz.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 2, item_saz)
            
            item_sel = QTableWidgetItem(f"{row['sun_el']:.1f}")
            item_sel.setFlags(item_sel.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 3, item_sel)
            
            # 4/5: Stow Az/El (Initially Copy Sun)
            self.table.setItem(r, 4, QTableWidgetItem(f"{row['sun_az']:.1f}"))
            self.table.setItem(r, 5, QTableWidgetItem(f"{row['sun_el']:.1f}"))
            
            # 6: Profile Safety (Initially Unknown/Calc)
            self.table.setItem(r, 6, QTableWidgetItem("❓"))
            
        self.table.blockSignals(False)
        self.table.selectRow(0)
        
        # Initial Calc for ALL rows (to populate Safety)
        # We do this for all rows so user sees status immediately
        # Might be slow for 1440 mins? 
        # InfiniteKernel is fast (0.1ms). 1440 * 0.1ms = 144ms. OK.
        for i in range(rows):
             self.update_row_safety(i, update_viz=(i==0))
        
    def on_time_slider(self, val):
        if val != self.current_idx:
            self.table.selectRow(val)
            
    def on_table_select(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows: return
        
        idx = rows[0].row()
        self.current_idx = idx
        
        self.sl_time.blockSignals(True)
        self.sl_time.setValue(idx)
        self.sl_time.blockSignals(False)
        
        self.update_ui_from_row(idx)
        
    def update_ui_from_row(self, idx):
        try:
            t = self.table.item(idx, 0).text()
            stow_az = float(self.table.item(idx, 4).text())
            stow_el = float(self.table.item(idx, 5).text())
            
            self.lbl_time.setText(t)
            
            self.sl_az.blockSignals(True)
            self.sl_el.blockSignals(True)
            self.sl_az.setValue(int(stow_az * 10))
            self.sl_el.setValue(int(stow_el * 10))
            self.lbl_az.setText(f"Stow Az: {stow_az:.1f}°")
            self.lbl_el.setText(f"Stow El: {stow_el:.1f}°")
            self.sl_az.blockSignals(False)
            self.sl_el.blockSignals(False)
            
            self.update_row_safety(idx)
            
        except ValueError:
            pass
            
    def on_teach_change(self):
        az = self.sl_az.value() / 10.0
        el = self.sl_el.value() / 10.0
        
        self.lbl_az.setText(f"Stow Az: {az:.1f}°")
        self.lbl_el.setText(f"Stow El: {el:.1f}°")
        
        self.table.blockSignals(True)
        self.table.item(self.current_idx, 4).setText(f"{az:.1f}")
        self.table.item(self.current_idx, 5).setText(f"{el:.1f}")
        self.table.blockSignals(False)
        
        self.update_row_safety(self.current_idx)
        
    def on_table_edit(self, row, col):
        if col in [4, 5]:
            self.update_ui_from_row(row)
            
    def update_row_safety(self, idx, update_viz=True):
        if not self.kernel: return
        
        try:
            stow_az = float(self.table.item(idx, 4).text())
            stow_el = float(self.table.item(idx, 5).text())
            
            row_data = self.day_data.iloc[idx]
            sun_az = row_data['sun_az']
            sun_el = row_data['sun_el']
            
            # Use negative plant rotation as per new coordinate system
            local_az = sun_az - (-self.state.config.plant_rotation)
            
            # Stow Angle is GLOBAL (from user input). Kernel needs LOCAL for Override.
            local_stow_az = stow_az - (-self.state.config.plant_rotation)
            
            # Solve with kernel
            # solve_timestep returns (states, collision_detected)
            # Collision=True means UNSAFE.
            states, collision = self.kernel.solve_timestep(
                local_az, sun_el,
                enable_safety=True,
                stow_override=(local_stow_az, stow_el)
            )
            
            is_safe = not collision
            
            # Update Table
            s_item = self.table.item(idx, 6)
            s_item.setText("✅" if is_safe else "❌")
            
            # Update Viz
            if update_viz and idx == self.current_idx:
                self.preview_update.emit(sun_az, sun_el, is_safe, states)
                
        except Exception as e:
            print(f"Safety Calc Error: {e}")
            
    def copy_and_next(self):
        if self.current_idx >= self.table.rowCount() - 1: return
        
        az = self.table.item(self.current_idx, 4).text()
        el = self.table.item(self.current_idx, 5).text()
        
        next_idx = self.current_idx + 1
        self.table.item(next_idx, 4).setText(az)
        self.table.item(next_idx, 5).setText(el)
        
        self.table.selectRow(next_idx)
        
    def reset_row(self):
        # Handle multiple selection
        rows = sorted(set(index.row() for index in self.table.selectedIndexes()))
        if not rows: rows = [self.current_idx]
        
        self.table.blockSignals(True)
        for idx in rows:
            az = self.table.item(idx, 2).text()
            el = self.table.item(idx, 3).text()
            self.table.item(idx, 4).setText(az)
            self.table.item(idx, 5).setText(el)
        self.table.blockSignals(False)
        
        self.update_ui_from_row(self.current_idx)
        # Update safety for all reset rows
        for idx in rows:
             self.update_row_safety(idx, update_viz=(idx==self.current_idx))

    def reset_all_rows(self):
        # Confirm?
        reply = QMessageBox.question(self, 'Reset Profile', 
             "Reset all stow angles to Sun Position?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No: return

        count = self.table.rowCount()
        self.table.blockSignals(True)
        for idx in range(count):
            az = self.table.item(idx, 2).text()
            el = self.table.item(idx, 3).text()
            self.table.item(idx, 4).setText(az)
            self.table.item(idx, 5).setText(el)
        self.table.blockSignals(False)
        
        # Update Safety
        for i in range(count):
             self.update_row_safety(i, update_viz=(i==self.current_idx))
             
        self.update_ui_from_row(self.current_idx)

    def interpolate_selection(self):
        rows = sorted(set(index.row() for index in self.table.selectedIndexes()))
        if len(rows) < 2: return
        
        start_r = rows[0]
        end_r = rows[-1]
        
        start_az = float(self.table.item(start_r, 4).text())
        start_el = float(self.table.item(start_r, 5).text())
        end_az = float(self.table.item(end_r, 4).text())
        end_el = float(self.table.item(end_r, 5).text())
        
        count = end_r - start_r
        for i in range(1, count):
            frac = i / count
            az = start_az + (end_az - start_az) * frac
            el = start_el + (end_el - start_el) * frac
            
            r = start_r + i
            self.table.item(r, 4).setText(f"{az:.1f}")
            self.table.item(r, 5).setText(f"{el:.1f}")
            self.update_row_safety(r)
            
    def generate_smooth_frames(self):
        """Generates interpolated frames between selected rows."""
        rows = sorted(set(index.row() for index in self.table.selectedIndexes()))
        if len(rows) < 2: 
             QMessageBox.warning(self, "Selection", "Select at least 2 rows.")
             return False
             
        self.smooth_frames = []
        STEPS = 10 
        
        # Pre-calc date for context (optional, just for viz)
        # We can iterate through the selection
        for i in range(len(rows) - 1):
             r1 = rows[i]
             r2 = rows[i+1]
             
             # Get Data
             t1 = self.table.item(r1, 0).text()
             az1 = float(self.table.item(r1, 2).text())
             el1 = float(self.table.item(r1, 3).text())
             saz1 = float(self.table.item(r1, 4).text())
             sel1 = float(self.table.item(r1, 5).text())

             t2 = self.table.item(r2, 0).text() # Unused for lerp but good for debug
             az2 = float(self.table.item(r2, 2).text())
             el2 = float(self.table.item(r2, 3).text())
             saz2 = float(self.table.item(r2, 4).text())
             sel2 = float(self.table.item(r2, 5).text())
             
             # Interpolate
             for j in range(STEPS):
                  frac = j / STEPS
                  
                  # Shortest Path Interpolation for Azimuth
                  # Diff range: -180 to 180
                  d_az = (az2 - az1 + 180) % 360 - 180
                  i_az = (az1 + d_az * frac) % 360
                  
                  i_el = el1 + (el2 - el1) * frac
                  
                  d_saz = (saz2 - saz1 + 180) % 360 - 180
                  i_saz = (saz1 + d_saz * frac) % 360
                  
                  i_sel = sel1 + (sel2 - sel1) * frac
                  
                  self.smooth_frames.append({
                       "sun_az": i_az,
                       "sun_el": i_el,
                       "stow_az": i_saz,
                       "stow_el": i_sel,
                       "label": f"Row {r1}->{r2} ({int(frac*100)}%)"
                  })
                  
        # Append final frame (Exact last row)
        last_r = rows[-1]
        self.smooth_frames.append({
             "sun_az": float(self.table.item(last_r, 2).text()),
             "sun_el": float(self.table.item(last_r, 3).text()),
             "stow_az": float(self.table.item(last_r, 4).text()),
             "stow_el": float(self.table.item(last_r, 5).text()),
             "label": f"Row {last_r} (End)"
        })
        
        return True

    def play_smooth(self):
        if self.is_playing:
             self.pause()
             # If play smooth clicked while playing normal, switch?
             # For simplicity, pause first.
             
        if not self.generate_smooth_frames(): return
        
        # Configure Slider
        self.sl_smooth.setEnabled(True)
        self.sl_smooth.blockSignals(True)
        self.sl_smooth.setRange(0, len(self.smooth_frames) - 1)
        self.sl_smooth.setValue(0)
        self.sl_smooth.blockSignals(False)
        
        self.smooth_idx = 0
        self.is_smooth_playing = True
        self.btn_play_smooth.setText("⏸ Stop Smooth")
        # Use faster timer?
        self.timer.setInterval(30) # 30ms ~ 33fps
        # Disconnect normal step
        try: self.timer.timeout.disconnect(self.step_forward)
        except: pass
        self.timer.timeout.connect(self.step_smooth)
        self.timer.start()
        
    def step_smooth(self):
        if self.smooth_idx < len(self.smooth_frames) - 1:
             self.smooth_idx += 1
             self.sl_smooth.setValue(self.smooth_idx) # Triggers update_smooth_viz
        else:
             self.stop_smooth()
             
    def stop_smooth(self):
        self.is_smooth_playing = False
        self.timer.stop()
        self.btn_play_smooth.setText("▶ Play Selected (Smooth)")
        try: self.timer.timeout.disconnect(self.step_smooth)
        except: pass
        # Reconnect normal
        self.timer.timeout.connect(self.step_forward)
        self.timer.setInterval(100) # Reset to 10fps
        
    def on_smooth_slider(self, val):
         self.smooth_idx = val
         self.update_smooth_viz(val)
         
    def update_smooth_viz(self, idx):
         if not hasattr(self, 'smooth_frames') or idx >= len(self.smooth_frames): return
         
         frame = self.smooth_frames[idx]
         
         # Calc Safety Realtime
         # Use negative plant rotation
         rot = -self.state.config.plant_rotation
         
         local_az = frame['sun_az'] - rot
         local_stow_az = frame['stow_az'] - rot
         
         states, collision = self.kernel.solve_timestep(
              local_az, frame['sun_el'], 
              enable_safety=False, # Viz: Always Track/Stow as requested
              stow_override=(local_stow_az, frame['stow_el'])
         )
         
         is_safe = not collision
         
         # Update Labels (Optional: Reuse Teach Labels or separate?)
         self.lbl_az.setText(f"Stow Az: {frame['stow_az']:.1f}° (Interpolated)")
         self.lbl_el.setText(f"Stow El: {frame['stow_el']:.1f}°")
         
         # Emit Viz Update
         self.preview_update.emit(frame['sun_az'], frame['sun_el'], is_safe, states)
            
    def play(self):
        self.is_playing = True
        self.timer.start()
        
    def pause(self):
        self.is_playing = False
        self.timer.stop()
        
    def step_forward(self):
        if self.current_idx < self.table.rowCount() - 1:
            self.table.selectRow(self.current_idx + 1)
        else:
            self.pause()

    def browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Profile Folder", self.txt_path.text())
        if d:
            self.txt_path.setText(d)
            self.refresh_profiles()

    def refresh_profiles(self):
        folder = self.txt_path.text()
        self.cb_profiles.clear()
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith(".json")]
            self.cb_profiles.addItems(sorted(files))

    def delete_profile(self):
        folder = self.txt_path.text()
        name = self.cb_profiles.currentText()
        if not name or not folder: return
        
        path = os.path.join(folder, name)
        reply = QMessageBox.question(self, 'Confirm Delete', f"Delete {name}?", 
             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                os.remove(path)
                self.refresh_profiles()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Delete failed: {e}")

    def save_profile(self):
        name = self.txt_name.text()
        folder = self.txt_path.text()
        if not name or not folder: return
        
        # Ensure dir
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except OSError:
                QMessageBox.critical(self, "Error", "Cannot create directory.")
                return
                
        # Prepare Config Dict (Exclude total_panels)
        cfg_dict = self.state.config.__dict__.copy() if hasattr(self.state.config, '__dict__') else {}
        if 'total_panels' in cfg_dict: del cfg_dict['total_panels']
        
        # Prepare Geo Dict (Fix PivotZ in metadata if desired, or just raw)
        geo_dict = self.state.geometry.__dict__.copy() if hasattr(self.state.geometry, '__dict__') else {}
        # User complained about saved pivot offset. 
        # We save exact internal state. If they want "Setup" value, we might calculate.
        # But saving internal state is safer for loading back. 
        # I'll keep raw internal but maybe comment or transform if strictly requested.
        # User said "pivot offset is again saved differently". They likely want 0.38 (config) vs 0.305 (internal).
        # We can reconstruct? pivot_z = pivot_offset[2] + thickness/2?
        # Let's save both for clarity or just force the Config Pivot Height.
        # If I change it here, loading logic must invert it.
        # Since I am just saving for Reference (likely), I will save a "display_pivot_z": val
        
        piv_z = geo_dict.get('pivot_offset', [0,0,0])[2] + (geo_dict.get('thickness', 0)/2.0)
        geo_dict['display_pivot_z_height'] = piv_z
        
        data = {
            "metadata": {
                "date_label": self.lbl_date.text(),
                "geometry": geo_dict,
                "config": cfg_dict
            },
            "rows": []
        }
        
        rows = self.table.rowCount()
        for r in range(rows):
            t_str = self.table.item(r, 0).text()
            sim_safety = self.table.item(r, 1).text() # Save Sim Safety Emoji/Status
            sun_az = float(self.table.item(r, 2).text())
            sun_el = float(self.table.item(r, 3).text())
            stow_az = float(self.table.item(r, 4).text())
            stow_el = float(self.table.item(r, 5).text())
            
            data["rows"].append({
                "time": t_str, 
                "sim_safety": sim_safety,
                "sun_az": sun_az,
                "sun_el": sun_el,
                "stow_az": stow_az, 
                "stow_el": stow_el
            })
            
        path = os.path.join(folder, f"{name}.json")
        if os.path.exists(path):
            reply = QMessageBox.question(self, 'Confirm Overwrite', f"File {name}.json exists. Overwrite?", 
                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No: return
            
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            QMessageBox.information(self, "Saved", f"Profile saved to {path}")
            self.refresh_profiles()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {e}")

    def load_profile(self):
        folder = self.txt_path.text()
        name = self.cb_profiles.currentText()
        if not name or not folder: return
        
        path = os.path.join(folder, name)
        if not os.path.exists(path):
             QMessageBox.warning(self, "Error", "File not found.")
             return
             
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.blockSignals(True) # Block signals during load
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            file_rows = data.get("rows", [])
            if not file_rows:
                return

            # --- No Interpolation Mode (Exploration) ---
            # 1. Reconstruct DataFrame directly from File
            # Columns needed for load_day_data: 'time', 'sun_az', 'sun_el', 'safety' (or sim_safety)
            
            new_df = pd.DataFrame(file_rows)
            
            # If 'time' is string, we keep it as is for display, but load_day_data expects specific handling?
            # load_day_data: 
            #   row['time'].strftime("%H:%M") -> assumes datetime object
            #   We need to convert 'time' back to datetime objects if possible, or adjust load_day_data.
            #   Let's try to convert. The date doesn't matter much for display (just HH:MM), 
            #   but for consistency let's use the date from metadata or today.
            
            meta_date = data.get("metadata", {}).get("date_label", "Unknown")
            
            try:
                # Try parsing. If format varies, might fail. 
                # Saved as "HH:MM" usually? 
                # Wait, in save_profile: t_str = self.table.item(r, 0).text() -> "HH:MM"
                # So the file contains generic string "HH:MM". 
                # PD.to_datetime will default to today or unix epoch.
                # simpler: Just ensure it's a datetime object so .strftime works, OR
                # Update load_day_data to handle string 'time'.
                new_df['time'] = pd.to_datetime(new_df['time'], format="%H:%M")
            except:
                # Fallback: if parse fails, maybe it's already string.
                # We need to monkey-patch or ensure load_day_data handles strings.
                pass
                
            # 2. Update Context Labels
            self.lbl_date.setText(f"Date: {meta_date} (Profile)")
            
            # 3. Reload Table
            # We reuse load_day_data to populate the table structure
            self.load_day_data(new_df, meta_date)
            
            # 4. Restore Stow Angles 
            # load_day_data overwrites cols 4/5 with Sun Position by default.
            # We need to overwrite them with the file's stow angles.
            # And since we passed new_df to load_day_data, it used new_df['sun_az']...
            # Does new_df have 'stow_az'? Yes.
            
            rows = len(new_df)
            self.table.blockSignals(True)
            for r in range(rows):
                row = new_df.iloc[r]
                # Overwrite the copied sun angles with actual stow angles
                if 'stow_az' in row and 'stow_el' in row:
                    self.table.item(r, 4).setText(f"{row['stow_az']:.1f}")
                    self.table.item(r, 5).setText(f"{row['stow_el']:.1f}")
            self.table.blockSignals(False)
            
            # 5. Recalc Safety (Profile Safety)
            # This uses the NEW self.day_data (which is new_df).
            # So it uses the FILE's sun position. This is what we want (Context Preservation).
            for i in range(rows):
                 self.update_row_safety(i, update_viz=(i==0))
                 
            self.update_ui_from_row(0)
            QMessageBox.information(self, "Loaded", f"Loaded profile {name} in exploration mode.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Load failed: {e}")
        finally:
            self.blockSignals(False)
            QApplication.restoreOverrideCursor()
