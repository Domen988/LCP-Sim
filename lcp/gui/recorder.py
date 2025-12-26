
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
                             QPushButton, QGroupBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QSplitter, QLineEdit, QComboBox, QMessageBox, 
                             QAbstractItemView, QFrame, QSizePolicy, QFileDialog, QApplication, QMenu, QScrollArea)
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

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

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
        # Set default profile folder and refresh
        import sys
        if getattr(sys, 'frozen', False):
            # If frozen, usage executable directory (external)
            app_root = os.path.dirname(sys.executable)
        else:
            # If dev, use project root relative to this file
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

    def set_viewport(self, viewport):
        self.viewport = viewport
        
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
        
        # 3. Teach INACTIVE (Stow Group)
        gb_teach_inaz = QGroupBox("Teach Inactive")
        teach_l = QVBoxLayout()
        
        # Azimuth
        self.lbl_in_az = QLabel("Inactive Az: 0.0°")
        self.sl_in_az = QSlider(Qt.Orientation.Horizontal)
        self.sl_in_az.setRange(0, 3600) # 0 to 360
        self.sl_in_az.valueChanged.connect(self.on_teach_change)
        
        # Elevation
        self.lbl_in_el = QLabel("Inactive El: 0.0°")
        self.sl_in_el = QSlider(Qt.Orientation.Horizontal)
        self.sl_in_el.setRange(0, 900)
        self.sl_in_el.valueChanged.connect(self.on_teach_change)
        
        self.btn_reset_inactive = QPushButton("Reset Inactive (to Sun)")
        self.btn_reset_inactive.clicked.connect(self.reset_inactive)
        
        teach_l.addWidget(self.lbl_in_az)
        teach_l.addWidget(self.sl_in_az)
        teach_l.addWidget(self.lbl_in_el)
        teach_l.addWidget(self.sl_in_el)
        teach_l.addWidget(self.btn_reset_inactive)
        gb_teach_inaz.setLayout(teach_l)
        l_layout.addWidget(gb_teach_inaz)
        
        # 4. Teach ACTIVE (Tracking Group)
        gb_teach_act = QGroupBox("Teach Active")
        act_teach_l = QVBoxLayout()
        
        # Azimuth
        self.lbl_act_az = QLabel("Active Az: 0.0°")
        self.sl_act_az = QSlider(Qt.Orientation.Horizontal)
        self.sl_act_az.setRange(0, 3600)
        self.sl_act_az.valueChanged.connect(self.on_teach_change)
        
        # Elevation
        self.lbl_act_el = QLabel("Active El: 0.0°")
        self.sl_act_el = QSlider(Qt.Orientation.Horizontal)
        self.sl_act_el.setRange(0, 900)
        self.sl_act_el.valueChanged.connect(self.on_teach_change)
        
        self.btn_reset_active = QPushButton("Reset Active (to Sun)")
        self.btn_reset_active.clicked.connect(self.reset_active)
        
        act_teach_l.addWidget(self.lbl_act_az)
        act_teach_l.addWidget(self.sl_act_az)
        act_teach_l.addWidget(self.lbl_act_el)
        act_teach_l.addWidget(self.sl_act_el)
        act_teach_l.addWidget(self.btn_reset_active)
        gb_teach_act.setLayout(act_teach_l)
        l_layout.addWidget(gb_teach_act)

        # 5. Actions
        gb_act = QGroupBox("Actions")
        act_l = QVBoxLayout()
        
        self.btn_copy_next = QPushButton("Copy & Next Row")
        self.btn_copy_next.clicked.connect(self.copy_and_next)
        
        h_interp = QHBoxLayout()
        self.btn_interp = QPushButton("Interpolate Selection")
        self.btn_interp.clicked.connect(self.interpolate_selection)
        self.btn_reset_row = QPushButton("Reset Row (Sun)")
        self.btn_reset_row.clicked.connect(self.reset_row)
        h_interp.addWidget(self.btn_interp)
        h_interp.addWidget(self.btn_reset_row)
        
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
        
        # 6. Export
        gb_exp = QGroupBox("Export")
        exp_l = QVBoxLayout()
        self.btn_export_gif = QPushButton("Export Selection as GIF")
        self.btn_export_gif.clicked.connect(self.export_gif)
        exp_l.addWidget(self.btn_export_gif)
        gb_exp.setLayout(exp_l)
        l_layout.addWidget(gb_exp)
        
        # 6. Profile Management
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
        
        file_l.addLayout(h_path)
        file_l.addLayout(h_name)
        file_l.addLayout(h_load)
        gb_file.setLayout(file_l)
        l_layout.addWidget(gb_file)
        
        l_layout.addStretch()
        
        # Wrap Controls in Scroll Area
        scroll_left = QScrollArea()
        scroll_left.setWidget(left_widget)
        scroll_left.setWidgetResizable(True)
        scroll_left.setFrameShape(QFrame.Shape.NoFrame)
        splitter.addWidget(scroll_left)
        
        # --- RIGHT: TABLE ---
        self.table = QTableWidget()
        # Cols: Time, Safety, SunAz, SunEl, InactiveAz, InactiveEl, ActiveAz, ActiveEl, ProfileSafety
        cols = ["Time", "Sim Safety", "Sun Az", "Sun El", 
                "Inactive Az", "Inactive El", "Active Az", "Active El", 
                "Profile Safety"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
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
            
            # 0: Time
            if hasattr(row['time'], 'strftime'):
                 time_str = row['time'].strftime("%H:%M")
            else:
                 time_str = str(row['time'])
                 
            item_t = QTableWidgetItem(time_str)
            item_t.setFlags(item_t.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 0, item_t)
            
            # 1: Sim Safety
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
            
            # 4/5: Inactive Az/El (Default: Sun)
            self.table.setItem(r, 4, QTableWidgetItem(f"{row['sun_az']:.1f}"))
            self.table.setItem(r, 5, QTableWidgetItem(f"{row['sun_el']:.1f}"))

            # 6/7: Active Az/El (Default: Sun)
            self.table.setItem(r, 6, QTableWidgetItem(f"{row['sun_az']:.1f}"))
            self.table.setItem(r, 7, QTableWidgetItem(f"{row['sun_el']:.1f}"))
            
            # 8: Profile Safety
            self.table.setItem(r, 8, QTableWidgetItem("❓"))
            
        self.table.blockSignals(False)
        self.table.selectRow(0)
        
        # Initial Calc
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
            in_az = float(self.table.item(idx, 4).text())
            in_el = float(self.table.item(idx, 5).text())
            act_az = float(self.table.item(idx, 6).text())
            act_el = float(self.table.item(idx, 7).text())
            
            self.lbl_time.setText(t)
            
            # Update Inactive
            self.sl_in_az.blockSignals(True)
            self.sl_in_el.blockSignals(True)
            self.sl_in_az.setValue(int(in_az * 10))
            self.sl_in_el.setValue(int(in_el * 10))
            self.lbl_in_az.setText(f"Inactive Az: {in_az:.1f}°")
            self.lbl_in_el.setText(f"Inactive El: {in_el:.1f}°")
            self.sl_in_az.blockSignals(False)
            self.sl_in_el.blockSignals(False)

            # Update Active
            self.sl_act_az.blockSignals(True)
            self.sl_act_el.blockSignals(True)
            self.sl_act_az.setValue(int(act_az * 10))
            self.sl_act_el.setValue(int(act_el * 10))
            self.lbl_act_az.setText(f"Active Az: {act_az:.1f}°")
            self.lbl_act_el.setText(f"Active El: {act_el:.1f}°")
            self.sl_act_az.blockSignals(False)
            self.sl_act_el.blockSignals(False)
            
            self.update_row_safety(idx)
            
        except ValueError:
            pass
            
    def on_teach_change(self):
        in_az = self.sl_in_az.value() / 10.0
        in_el = self.sl_in_el.value() / 10.0
        act_az = self.sl_act_az.value() / 10.0
        act_el = self.sl_act_el.value() / 10.0
        
        self.lbl_in_az.setText(f"Inactive Az: {in_az:.1f}°")
        self.lbl_in_el.setText(f"Inactive El: {in_el:.1f}°")
        self.lbl_act_az.setText(f"Active Az: {act_az:.1f}°")
        self.lbl_act_el.setText(f"Active El: {act_el:.1f}°")
        
        self.table.blockSignals(True)
        self.table.item(self.current_idx, 4).setText(f"{in_az:.1f}")
        self.table.item(self.current_idx, 5).setText(f"{in_el:.1f}")
        self.table.item(self.current_idx, 6).setText(f"{act_az:.1f}")
        self.table.item(self.current_idx, 7).setText(f"{act_el:.1f}")
        self.table.blockSignals(False)
        
        self.update_row_safety(self.current_idx)
        
    def reset_inactive(self):
        # Reset current Inactive to Sun
        sun_az = float(self.table.item(self.current_idx, 2).text())
        sun_el = float(self.table.item(self.current_idx, 3).text())
        
        self.sl_in_az.setValue(int(sun_az * 10))
        self.sl_in_el.setValue(int(sun_el * 10))
        # Trigger checked by signal
        
    def reset_active(self):
        # Reset current Active to Sun
        sun_az = float(self.table.item(self.current_idx, 2).text())
        sun_el = float(self.table.item(self.current_idx, 3).text())
        
        self.sl_act_az.setValue(int(sun_az * 10))
        self.sl_act_el.setValue(int(sun_el * 10))

    def on_table_edit(self, row, col):
        if col in [4, 5, 6, 7]:
            self.update_ui_from_row(row)

    def show_context_menu(self, position):
        menu = QMenu()
        copy_action = menu.addAction("Copy Selection")
        action = menu.exec(self.table.viewport().mapToGlobal(position))
        
        if action == copy_action:
            self.copy_selection_to_clipboard()

    def copy_selection_to_clipboard(self):
        selection = self.table.selectedRanges()
        if not selection: return
        
        # Get bounds of selection
        # QTableWidget selection can be multiple separate ranges, but typically users 
        # drag-select a contiguous block. We'll handle multiple ranges by concatenating?
        # Or just take top-left to bottom-right of all selected items.
        
        # Simpler approach: iterate all selected items and sort them by row/col
        indexes = self.table.selectedIndexes()
        if not indexes: return
        
        rows = sorted(set(index.row() for index in indexes))
        cols = sorted(set(index.column() for index in indexes))
        
        # Create CSV/TSV string
        # Use TSV for Excel compatibility
        text_data = ""
        
        # Header (Optional? Maybe skip header for copy-paste inside data)
        # Usually users want just data.
        
        for r in rows:
            row_data = []
            for c in cols:
                item = self.table.item(r, c)
                txt = item.text() if item else ""
                row_data.append(txt)
            text_data += "\t".join(row_data) + "\n"
            
        QApplication.clipboard().setText(text_data)
        # QMessageBox.information(self, "Copied", "Selection copied to clipboard.") # Optional feedback (annoying?)
            
    def update_row_safety(self, idx, update_viz=True):
        if not self.kernel: return
        
        try:
            in_az = float(self.table.item(idx, 4).text())
            in_el = float(self.table.item(idx, 5).text())
            act_az = float(self.table.item(idx, 6).text())
            act_el = float(self.table.item(idx, 7).text())
            
            row_data = self.day_data.iloc[idx]
            sun_az = row_data['sun_az']
            sun_el = row_data['sun_el']
            
            # Conversion to Local Plant Coordinates
            rot = -self.state.config.plant_rotation
            local_sun_az = sun_az - rot
            local_in_az = in_az - rot
            local_act_az = act_az - rot
            
            # Solve
            states, collision = self.kernel.solve_timestep(
                local_sun_az, sun_el,
                enable_safety=True,
                inactive_override=(local_in_az, in_el),
                active_override=(local_act_az, act_el)
            )
            
            is_safe = not collision
            
            # Update Table
            s_item = self.table.item(idx, 8)
            s_item.setText("✅" if is_safe else "❌")
            
            # Update Viz
            if update_viz and idx == self.current_idx:
                self.preview_update.emit(sun_az, sun_el, is_safe, states)
                
        except Exception as e:
            print(f"Safety Calc Error: {e}")
            
    def copy_and_next(self):
        if self.current_idx >= self.table.rowCount() - 1: return
        
        in_az = self.table.item(self.current_idx, 4).text()
        in_el = self.table.item(self.current_idx, 5).text()
        act_az = self.table.item(self.current_idx, 6).text()
        act_el = self.table.item(self.current_idx, 7).text()
        
        next_idx = self.current_idx + 1
        self.table.item(next_idx, 4).setText(in_az)
        self.table.item(next_idx, 5).setText(in_el)
        self.table.item(next_idx, 6).setText(act_az)
        self.table.item(next_idx, 7).setText(act_el)
        
        self.table.selectRow(next_idx)
        
    def reset_row(self):
        # Handle multiple selection
        rows = sorted(set(index.row() for index in self.table.selectedIndexes()))
        if not rows: rows = [self.current_idx]
        
        self.table.blockSignals(True)
        for idx in rows:
            az = self.table.item(idx, 2).text()
            el = self.table.item(idx, 3).text()
            # Set Inactive
            self.table.item(idx, 4).setText(az)
            self.table.item(idx, 5).setText(el)
            # Set Active
            self.table.item(idx, 6).setText(az)
            self.table.item(idx, 7).setText(el)
        self.table.blockSignals(False)
        
        self.update_ui_from_row(self.current_idx)
        # Update safety for all reset rows
        for idx in rows:
             self.update_row_safety(idx, update_viz=(idx==self.current_idx))

    def reset_all_rows(self):
        # Confirm?
        reply = QMessageBox.question(self, 'Reset Profile', 
             "Reset all angles to Sun Position?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No: return

        count = self.table.rowCount()
        self.table.blockSignals(True)
        for idx in range(count):
            az = self.table.item(idx, 2).text()
            el = self.table.item(idx, 3).text()
            
            self.table.item(idx, 4).setText(az)
            self.table.item(idx, 5).setText(el)
            self.table.item(idx, 6).setText(az)
            self.table.item(idx, 7).setText(el)
            
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
        
        s_in_az = float(self.table.item(start_r, 4).text())
        s_in_el = float(self.table.item(start_r, 5).text())
        e_in_az = float(self.table.item(end_r, 4).text())
        e_in_el = float(self.table.item(end_r, 5).text())

        s_act_az = float(self.table.item(start_r, 6).text())
        s_act_el = float(self.table.item(start_r, 7).text())
        e_act_az = float(self.table.item(end_r, 6).text())
        e_act_el = float(self.table.item(end_r, 7).text())
        
        count = end_r - start_r
        for i in range(1, count):
            frac = i / count
            
            # Inactive
            i_az = s_in_az + (e_in_az - s_in_az) * frac
            i_el = s_in_el + (e_in_el - s_in_el) * frac
            
            # Active
            a_az = s_act_az + (e_act_az - s_act_az) * frac
            a_el = s_act_el + (e_act_el - s_act_el) * frac
            
            r = start_r + i
            self.table.item(r, 4).setText(f"{i_az:.1f}")
            self.table.item(r, 5).setText(f"{i_el:.1f}")
            self.table.item(r, 6).setText(f"{a_az:.1f}")
            self.table.item(r, 7).setText(f"{a_el:.1f}")
            self.update_row_safety(r)
            
    def generate_smooth_frames(self):
        """Generates interpolated frames between selected rows."""
        rows = sorted(set(index.row() for index in self.table.selectedIndexes()))
        if len(rows) < 2: 
             QMessageBox.warning(self, "Selection", "Select at least 2 rows.")
             return False
             
        self.smooth_frames = []
        STEPS = 10 
        
        for i in range(len(rows) - 1):
             r1 = rows[i]
             r2 = rows[i+1]
             
             # Get Data 1
             az1 = float(self.table.item(r1, 2).text())
             el1 = float(self.table.item(r1, 3).text())
             in_az1 = float(self.table.item(r1, 4).text())
             in_el1 = float(self.table.item(r1, 5).text())
             act_az1 = float(self.table.item(r1, 6).text())
             act_el1 = float(self.table.item(r1, 7).text())

             # Get Data 2
             az2 = float(self.table.item(r2, 2).text())
             el2 = float(self.table.item(r2, 3).text())
             in_az2 = float(self.table.item(r2, 4).text())
             in_el2 = float(self.table.item(r2, 5).text())
             act_az2 = float(self.table.item(r2, 6).text())
             act_el2 = float(self.table.item(r2, 7).text())
             
             # Interpolate
             for j in range(STEPS):
                  frac = j / STEPS
                  
                  # Sun
                  d_az = (az2 - az1 + 180) % 360 - 180
                  s_az = (az1 + d_az * frac) % 360
                  s_el = el1 + (el2 - el1) * frac
                  
                  # Inactive
                  d_iaz = (in_az2 - in_az1 + 180) % 360 - 180
                  i_az = (in_az1 + d_iaz * frac) % 360
                  i_el = in_el1 + (in_el2 - in_el1) * frac

                  # Active
                  d_aaz = (act_az2 - act_az1 + 180) % 360 - 180
                  a_az = (act_az1 + d_aaz * frac) % 360
                  a_el = act_el1 + (act_el2 - act_el1) * frac
                  
                  self.smooth_frames.append({
                       "sun_az": s_az,
                       "sun_el": s_el,
                       "inactive_az": i_az,
                       "inactive_el": i_el,
                       "active_az": a_az,
                       "active_el": a_el,
                       "label": f"Row {r1}->{r2}"
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
         rot = -self.state.config.plant_rotation
         
         local_sun_az = frame['sun_az'] - rot
         local_in_az = frame['inactive_az'] - rot
         local_act_az = frame['active_az'] - rot
         
         states, collision = self.kernel.solve_timestep(
              local_sun_az, frame['sun_el'], 
              enable_safety=False,
              inactive_override=(local_in_az, frame['inactive_el']),
              active_override=(local_act_az, frame['active_el'])
         )
         
         is_safe = not collision
         
         # Update Labels
         self.lbl_in_az.setText(f"Inactive Az: {frame['inactive_az']:.1f}° (Sim)")
         self.lbl_act_az.setText(f"Active Az: {frame['active_az']:.1f}° (Sim)")
         
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
            in_az = float(self.table.item(r, 4).text())
            in_el = float(self.table.item(r, 5).text())
            act_az = float(self.table.item(r, 6).text())
            act_el = float(self.table.item(r, 7).text())
            
            data["rows"].append({
                "time": t_str, 
                "sim_safety": sim_safety,
                "sun_az": sun_az,
                "sun_el": sun_el,
                "inactive_az": in_az, 
                "inactive_el": in_el,
                "active_az": act_az,
                "active_el": act_el
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
            
            new_df = pd.DataFrame(file_rows)
            meta_date = data.get("metadata", {}).get("date_label", "Unknown")
            
            try:
                new_df['time'] = pd.to_datetime(new_df['time'], format="%H:%M")
            except:
                pass
                
            # 2. Update Context Labels
            self.lbl_date.setText(f"Date: {meta_date} (Profile)")
            
            # 3. Reload Table
            # We reuse load_day_data to populate the table structure
            self.load_day_data(new_df, meta_date)
            
            # 4. Restore Stow Angles 
            # load_day_data overwrites vals with Sun Position by default.
            # We need to overwrite them with the file's stow angles.
            
            rows = len(new_df)
            self.table.blockSignals(True)
            for r in range(rows):
                row = new_df.iloc[r]
                
                # Default logic: Active/Inactive are reset to sun if not found
                # But we just called load_day_data which set them to sun.
                
                # Check for Legacy Keys or New Keys
                in_az = None; in_el = None
                act_az = None; act_el = None
                
                # Inactive
                if 'inactive_az' in row: in_az = row['inactive_az']
                elif 'stow_az' in row: in_az = row['stow_az'] # Legacy
                
                if 'inactive_el' in row: in_el = row['inactive_el']
                elif 'stow_el' in row: in_el = row['stow_el'] # Legacy
                
                # Active
                if 'active_az' in row: act_az = row['active_az']
                if 'active_el' in row: act_el = row['active_el']
                
                # Write to Table if found
                if in_az is not None: self.table.item(r, 4).setText(f"{in_az:.1f}")
                if in_el is not None: self.table.item(r, 5).setText(f"{in_el:.1f}")
                
                if act_az is not None: self.table.item(r, 6).setText(f"{act_az:.1f}")
                if act_el is not None: self.table.item(r, 7).setText(f"{act_el:.1f}")
                
            self.table.blockSignals(False)
            
            # 5. Recalc Safety (Profile Safety)
            for i in range(rows):
                 self.update_row_safety(i, update_viz=(i==0))
                 
            self.update_ui_from_row(0)
            QMessageBox.information(self, "Loaded", f"Loaded profile {name} in exploration mode.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Load failed: {e}")
        finally:
            self.blockSignals(False)
            QApplication.restoreOverrideCursor()

    def export_gif(self):
        if not HAS_PIL:
            QMessageBox.critical(self, "Error", "PIL (Pillow) library not found.\nPlease install it to use this feature.")
            return

        if not hasattr(self, 'viewport') or not self.viewport:
            QMessageBox.critical(self, "Error", "Viewport not linked via MainWindow.")
            return

        # Ensure frames exist (User selection flow)
        if not self.generate_smooth_frames(): return
        
        path, _ = QFileDialog.getSaveFileName(self, "Save GIF Animation", "", "GIF Files (*.gif)")
        if not path: return
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            images = []
            
            # Loop
            for i in range(len(self.smooth_frames)):
                # Update State
                self.update_smooth_viz(i)
                QApplication.processEvents() # Force Redraw
                
                # Capture
                qimg = self.viewport.grabFramebuffer()
                
                # Convert QImage -> PIL
                qimg = qimg.convertToFormat(qimg.Format.Format_RGBA8888)
                w, h = qimg.width(), qimg.height()
                ptr = qimg.bits()
                ptr.setsize(qimg.sizeInBytes())
                
                raw = bytes(ptr)
                img = Image.frombytes("RGBA", (w, h), raw)
                images.append(img)
                
            # Save
            if images:
                # durations is in ms. We used 30ms timer ~ 33fps.
                images[0].save(path, save_all=True, append_images=images[1:], duration=33, loop=0)
                QMessageBox.information(self, "Success", f"GIF exported to {path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
        finally:
            QApplication.restoreOverrideCursor()
