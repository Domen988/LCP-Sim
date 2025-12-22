
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
                             QPushButton, QGroupBox, QLineEdit, QComboBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from datetime import datetime, timedelta

from lcp.gui.state import AppState
from lcp.core.stow import StowProfile
from lcp.core.solar import SolarCalculator

class StowRecorder(QWidget):
    """
    Bottom Dock Widget for Timeline Control and 'Teach Mode'.
    """
    
    # Signals
    scene_update = pyqtSignal() # Time changed or Manual Override
    
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.profile = StowProfile()
        self.solar = SolarCalculator()
        
        # Internal State
        self.current_time = datetime(2025, 1, 1, 12, 0, 0)
        self.is_playing = False
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.on_tick)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        # --- COL 1: Timeline ---
        gb_time = QGroupBox("Timeline")
        l_time = QVBoxLayout()
        
        # Display
        self.lbl_time = QLabel()
        self.lbl_time.setStyleSheet("font-size: 16pt; font-family: monospace; font-weight: bold;")
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.update_time_label()
        
        # Controls
        h_ctrl = QHBoxLayout()
        btn_prev = QPushButton("◀ -10m")
        btn_prev.clicked.connect(lambda: self.step_time(-10))
        
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self.toggle_play)
        
        btn_next = QPushButton("+10m ▶")
        btn_next.clicked.connect(lambda: self.step_time(10))
        
        h_ctrl.addWidget(btn_prev)
        h_ctrl.addWidget(self.btn_play)
        h_ctrl.addWidget(btn_next)
        
        l_time.addWidget(self.lbl_time)
        l_time.addLayout(h_ctrl)
        gb_time.setLayout(l_time)
        layout.addWidget(gb_time, stretch=1)
        
        # --- COL 2: Manual Override ---
        gb_man = QGroupBox("Manual Stow Teach")
        l_man = QVBoxLayout()
        
        # Sliders
        # Azimuth
        self.lbl_stow_az = QLabel(f"Azimuth: {self.state.stow_az:.1f}")
        self.sl_stow_az = QSlider(Qt.Orientation.Horizontal)
        self.sl_stow_az.setRange(-1800, 1800)
        self.sl_stow_az.setValue(int(self.state.stow_az * 10))
        self.sl_stow_az.valueChanged.connect(self.on_manual_change)
        
        # Elevation
        self.lbl_stow_el = QLabel(f"Elevation: {self.state.stow_el:.1f}")
        self.sl_stow_el = QSlider(Qt.Orientation.Horizontal)
        self.sl_stow_el.setRange(0, 900)
        self.sl_stow_el.setValue(int(self.state.stow_el * 10))
        self.sl_stow_el.valueChanged.connect(self.on_manual_change)
        
        # Record Button
        btn_rec = QPushButton("🔴 RECORD KEYFRAME")
        btn_rec.setStyleSheet("color: red; font-weight: bold;")
        btn_rec.clicked.connect(self.record_keyframe)
        
        l_man.addWidget(self.lbl_stow_az)
        l_man.addWidget(self.sl_stow_az)
        l_man.addWidget(self.lbl_stow_el)
        l_man.addWidget(self.sl_stow_el)
        l_man.addWidget(btn_rec)
        
        # Mode Selection?
        # TODO: Add "Playback Mode" selector (Manual vs Play Profile)
        
        gb_man.setLayout(l_man)
        layout.addWidget(gb_man, stretch=2)
        
    def update_time_label(self):
        self.lbl_time.setText(self.current_time.strftime("%Y-%m-%d\n%H:%M"))
        # Update Sun State
        sun = self.solar.get_position(self.current_time)
        self.state.sun_az = sun.azimuth
        self.state.sun_el = sun.elevation
        # Note: We must emit signal to trigger repaint
        
    def step_time(self, minutes):
        self.current_time += timedelta(minutes=minutes)
        self.update_time_label()
        
        # If in "Play Profile" mode, we should interpolate profile and set stow_az/el
        # For now, Manual dominance or interpolate? 
        # Feature Parity: If playing, update sliders.
        pos = self.profile.get_position_at(self.current_time)
        if pos:
            az, el = pos
            # Update Sliders (Signal blocked?)
            self.sl_stow_az.blockSignals(True)
            self.sl_stow_el.blockSignals(True)
            self.sl_stow_az.setValue(int(az*10))
            self.sl_stow_el.setValue(int(el*10))
            self.sl_stow_az.blockSignals(False)
            self.sl_stow_el.blockSignals(False)
            
            self.state.stow_az = az
            self.state.stow_el = el
            
            self.lbl_stow_az.setText(f"Azimuth: {az:.1f}")
            self.lbl_stow_el.setText(f"Elevation: {el:.1f}")
            
        self.scene_update.emit()

    def on_manual_change(self):
        az = self.sl_stow_az.value() / 10.0
        el = self.sl_stow_el.value() / 10.0
        
        self.state.stow_az = az
        self.state.stow_el = el
        
        self.lbl_stow_az.setText(f"Azimuth: {az:.1f}")
        self.lbl_stow_el.setText(f"Elevation: {el:.1f}")
        
        self.scene_update.emit()
        
    def record_keyframe(self):
        self.profile.add_keyframe(self.current_time, self.state.stow_az, self.state.stow_el)
        # Feedback? Tooltip or Status Bar. 
        # MainWindow status bar.
        
    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.play_timer.stop()
            self.btn_play.setText("▶ Play")
        else:
            self.is_playing = True
            self.play_timer.start(50) # 20fps logic update
            self.btn_play.setText("⏸ Pause")
            
    def on_tick(self):
        self.step_time(10) # Fast forward
