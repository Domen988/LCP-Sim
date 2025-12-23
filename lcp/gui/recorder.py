
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
            time_str = row['time'].strftime("%H:%M")
            # Sim Safety: row['safety'] is Collision Boolean (True=Clash)
            # We want: Safe=Check, Clash=X
            # If row['safety'] (Collision) is True -> Clash -> X
            safety = "❌" if row['safety'] else "✅"
            
            # 0: Time
            item_t = QTableWidgetItem(time_str)
            item_t.setFlags(item_t.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 0, item_t)
            
            # 1: Sim Safety
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
            
            local_az = sun_az - self.state.plant_rotation
            
            # Stow Angle is GLOBAL (from user input). Kernel needs LOCAL for Override.
            local_stow_az = stow_az - self.state.plant_rotation
            
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
            sun_az = float(self.table.item(r, 2).text())
            sun_el = float(self.table.item(r, 3).text())
            stow_az = float(self.table.item(r, 4).text())
            stow_el = float(self.table.item(r, 5).text())
            
            data["rows"].append({
                "time": t_str, 
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
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            file_rows = data.get("rows", [])
            if not file_rows:
                return

            # --- Interpolation Logic ---
            # 1. Extract File Data (Time, StowAz, StowEl)
            # We assume file rows are sorted by time.
            # Convert time strings to seconds from start of day (or similar metric) for robust interpolation?
            # Or just use the 'time' string if it is standard iso? No, need numeric for interp.
            # The 'time' in rows is likely a string.
            
            # Helper to parse time to seconds
            def parse_time_str(t_str):
                 # Handle "YYYY-MM-DDTHH:MM:SS" or similar
                 # Simpler: If the file has 'time', parse it.
                 try:
                     return pd.to_datetime(t_str).timestamp()
                 except:
                     return 0.0

            # Gather valid stow points from file
            f_times = []
            f_az = []
            f_el = []
            
            for r in file_rows:
                if 'stow_az' in r and 'stow_el' in r and 'time' in r:
                     f_times.append(parse_time_str(r['time']))
                     f_az.append(float(r['stow_az']))
                     f_el.append(float(r['stow_el']))
            
            if not f_times:
                QMessageBox.warning(self, "Warning", "Profile contains no valid stow data.")
                return

            # 2. Get Target Times from Table
            # Table timestamps are in current_day_data DataFrame usually, or we can parse column 0
            # Ideally use the DataFrame self.day_data if available
            if self.day_data is None: 
                 return

            # Use pandas generic time conversion
            target_times_pd = pd.to_datetime(self.day_data['time'])
            target_times = target_times_pd.map(lambda x: x.timestamp()).to_numpy()
            
            # 3. Interpolate
            # If file has only 1 point, fill all.
            if len(f_times) == 1:
                interp_az = np.full(len(target_times), f_az[0])
                interp_el = np.full(len(target_times), f_el[0])
            else:
                interp_az = np.interp(target_times, f_times, f_az)
                interp_el = np.interp(target_times, f_times, f_el)

            self.table.blockSignals(True)
            for i in range(len(target_times)):
                self.table.item(i, 4).setText(f"{interp_az[i]:.1f}")
                self.table.item(i, 5).setText(f"{interp_el[i]:.1f}")
            self.table.blockSignals(False)
            
            # Recalc Safety
            for i in range(len(target_times)):
                 self.update_row_safety(i, update_viz=(i==self.current_idx))
                 
            self.update_ui_from_row(self.current_idx)
            QMessageBox.information(self, "Loaded", f"Loaded profile {name} (interpolated to current timeline).")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Load failed: {e}")
        finally:
            QApplication.restoreOverrideCursor()
