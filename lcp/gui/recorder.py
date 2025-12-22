
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
                             QPushButton, QGroupBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QFileDialog, QCheckBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from datetime import datetime, timedelta
import pandas as pd

from lcp.gui.state import AppState
from lcp.core.stow import StowProfile, Keyframe
from lcp.core.solar import SolarCalculator

class StowRecorder(QWidget):
    """
    Bottom Dock Widget for 'Teach Mode' tightly integrated with Results Replay.
    It receives the current time from the Simulation Replay.
    """
    
    # Signals
    # Emits when Manual Sliders change (Panel Position Override)
    # args: az, el
    manual_override = pyqtSignal(float, float)
    
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        
        # Ensure Shared Profile
        if self.state.stow_profile is None:
             from lcp.core.stow import StowProfile
             self.state.stow_profile = StowProfile()
        self.profile = self.state.stow_profile
        
        self.solar = SolarCalculator()
        
        # Internal State
        # Default Start Time: Noon Jan 1st
        self.current_time = datetime(2025, 1, 1, 12, 0, 0)
        self.trace_active = False # If True, Stowed panels track the sun (same as active)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        # --- COL 1: Trace Mode ---
        gb_t = QGroupBox("Mode")
        l_t = QVBoxLayout()
        
        self.lbl_time = QLabel("--:--")
        self.lbl_time.setFont(QFont("Monospace", 12, QFont.Weight.Bold))
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.cb_trace = QCheckBox("Trace in this timestep")
        self.cb_trace.setToolTip("Stowed panels take the same position as active panels")
        self.cb_trace.toggled.connect(self.on_trace_toggle)
        
        l_t.addWidget(self.lbl_time)
        l_t.addWidget(self.cb_trace)
        gb_t.setLayout(l_t)
        layout.addWidget(gb_t, stretch=1)
        
        # --- COL 2: Stow Position ---
        gb_man = QGroupBox("Stow Position (Teach)")
        l_man = QVBoxLayout()
        
        # Sliders
        # Azimuth
        self.lbl_stow_az = QLabel(f"Azimuth: {self.state.stow_az:.1f}°")
        self.sl_stow_az = QSlider(Qt.Orientation.Horizontal)
        self.sl_stow_az.setRange(-1800, 1800) # -180 to 180
        self.sl_stow_az.setValue(int(self.state.stow_az * 10))
        self.sl_stow_az.valueChanged.connect(self.on_manual_change)
        
        # Elevation
        self.lbl_stow_el = QLabel(f"Elevation: {self.state.stow_el:.1f}°")
        self.sl_stow_el = QSlider(Qt.Orientation.Horizontal)
        self.sl_stow_el.setRange(0, 900) # 0 to 90
        self.sl_stow_el.setValue(int(self.state.stow_el * 10))
        self.sl_stow_el.valueChanged.connect(self.on_manual_change)
        
        # Record Button
        h_rec = QHBoxLayout()
        btn_rec = QPushButton("🔴 RECORD KEYFRAME")
        btn_rec.setStyleSheet("color: red; font-weight: bold;")
        btn_rec.clicked.connect(self.record_keyframe)
        h_rec.addWidget(btn_rec)
        
        l_man.addWidget(self.lbl_stow_az)
        l_man.addWidget(self.sl_stow_az)
        l_man.addWidget(self.lbl_stow_el)
        l_man.addWidget(self.sl_stow_el)
        l_man.addLayout(h_rec)
        
        gb_man.setLayout(l_man)
        layout.addWidget(gb_man, stretch=2)
        
        # --- COL 3: Keyframe List ---
        gb_list = QGroupBox("Keyframes")
        l_list = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Time", "Az", "El"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        l_list.addWidget(self.table)
        
        # Save/Load
        h_file = QHBoxLayout()
        btn_save = QPushButton("Save")
        btn_save.clicked.connect(self.save_profile)
        btn_load = QPushButton("Load")
        btn_load.clicked.connect(self.load_profile)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self.clear_profile)
        
        h_file.addWidget(btn_save)
        h_file.addWidget(btn_load)
        h_file.addWidget(btn_clear)
        l_list.addLayout(h_file)
        
        gb_list.setLayout(l_list)
        layout.addWidget(gb_list, stretch=2)
        
    def set_current_time(self, dt: datetime):
        """Called by ResultsWidget (via MainWindow) when replay time changes."""
        self.current_time = dt
        self.lbl_time.setText(dt.strftime("%d %H:%M"))
        
        # If Trace Mode is ON, calculate Start Tracking angle and update sliders
        if self.trace_active:
             sun = self.solar.get_position(dt)
             
             # Calculate Tracking Angles (Simplified for now - assumes Backtracking is part of Sim, 
             # but here we just want 'Active' position).
             # To match exact active panel position, we strictly need physics engine output.
             # However, assuming 'Active' means 'Tracking Sun' without backtracking:
             # Just point at sun.
             # If we want detailed tracking angle, we'd need access to Kinematics.
             # Let's use Sun Az/El as proxy for now, but Azimuth must be offset by plant rotation.
             # Note: SolarCalculator returns True North Azimuth.
             
             # BUT: The 'Active Panels' in simulation use backtracking.
             # If we want exact match, we should probably ASK the simulation?
             # But 'set_current_time' just gets time.
             # Simpler approach: Just point at Sun (Ideal Tracking).
             
             target_az = sun.azimuth
             target_el = max(0, sun.elevation)
             
             # Update Sliders (signals will emit manual_override)
             self.sl_stow_az.blockSignals(True)
             self.sl_stow_el.blockSignals(True)
             self.sl_stow_az.setValue(int(target_az * 10))
             self.sl_stow_el.setValue(int(target_el * 10))
             self.sl_stow_az.blockSignals(False)
             self.sl_stow_el.blockSignals(False)
             
             # Manually trigger updates? No, if we just set sliders, we don't emit.
             # We want the 'stowed' panels to move to this position visually.
             # So we must emit manual_override.
             self.on_manual_change()
             
    def on_trace_toggle(self):
        self.trace_active = self.cb_trace.isChecked()
        self.sl_stow_az.setEnabled(not self.trace_active)
        self.sl_stow_el.setEnabled(not self.trace_active)
        
        if self.trace_active:
             # Trigger update immediately
             self.set_current_time(self.current_time)

    def on_manual_change(self):
        az = self.sl_stow_az.value() / 10.0
        el = self.sl_stow_el.value() / 10.0
        
        self.state.stow_az = az
        self.state.stow_el = el
        
        self.lbl_stow_az.setText(f"Azimuth: {az:.1f}°")
        self.lbl_stow_el.setText(f"Elevation: {el:.1f}°")
        
        # Force Viewport Update
        self.manual_override.emit(az, el)
        
    def record_keyframe(self):
        self.profile.add_keyframe(self.current_time, self.state.stow_az, self.state.stow_el)
        self.refresh_table()
        
    def refresh_table(self):
        self.table.setRowCount(0)
        self.table.setRowCount(len(self.profile.keyframes))
        for i, k in enumerate(self.profile.keyframes):
            self.table.setItem(i, 0, QTableWidgetItem(k.timestamp.strftime("%Y-%m-%d %H:%M")))
            self.table.setItem(i, 1, QTableWidgetItem(f"{k.az:.1f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{k.el:.1f}"))
            
    def save_profile(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Stow Profile", "", "JSON Files (*.json)")
        if path:
            self.profile.save(path)
            
    def load_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Stow Profile", "", "JSON Files (*.json)")
        if path:
            self.profile = StowProfile.load(path)
            self.refresh_table()
            
    def clear_profile(self):
        self.profile.keyframes = []
        self.refresh_table()
