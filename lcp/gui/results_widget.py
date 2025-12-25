
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QPushButton, QSlider, QLabel, 
                             QSplitter, QFrame, QGridLayout, QGroupBox, QApplication)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QFont
import pyqtgraph as pg
import pandas as pd
import numpy as np
from datetime import datetime

# Physics Imports
from lcp.physics.engine import InfiniteKernel, PanelState
from lcp.gui.landscape_widget import AnnualLandscapeWidget
from lcp.gui.recorder import StowRecorder
from lcp.app.theme import Theme

class ResultsWidget(QWidget):
    # Signals
    # args: sun_az, sun_el, safely_detected, states_list
    replay_frame = pyqtSignal(float, float, bool, list)
    # args: current_datetime
    time_changed = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Internal State
        self.all_data = [] 
        self.current_day_data = None
        self.current_frames = []
        
        self.replay_idx = 0
        self.is_playing = False
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.timer.setInterval(50) # 20 FPS
        
        self.setup_ui()
        
        self.enable_safety = False # Default off per Sidebar UI
        
    @property
    def state(self):
        return self._state
        
    @state.setter
    def state(self, val):
        self._state = val
        # Retrieve needed references for Recorder
        # Initialize Recorder Tab now that we have State
        if not hasattr(self, 'recorder') or self.recorder is None:
             self.recorder = StowRecorder(self.state)
             self.tabs.addTab(self.recorder, "Stow Recorder")
             
             # Ensure Kernel Exists
             if not hasattr(self, 'kernel') or self.kernel is None:
                  self.kernel = InfiniteKernel(self.state.geometry, self.state.config)
                  
             # Inject the Kernel into Recorder (We share the one from ResultsWidget)
             self.recorder.set_kernel(self.kernel)
             
             # CONNECT SIGNAL: Recorder -> Viewport Update
             # We need to bridge this signal to the outside (MainWindow)
             # Let's define a new signal on ResultsWidget or reuse?
             # We can define `recorder_update` signal on ResultsWidget and emit it
             self.recorder.preview_update.connect(self.on_recorder_preview)

    def on_recorder_preview(self, az, el, safety, states):
         # Re-emit to MainWindow via existing signal?
         # `replay_frame` signature: (float, float, bool, list)
         # This matches!
         self.replay_frame.emit(az, el, safety, states)
         
    def on_create_stow_profile(self):
        # 1. Get Selected Day Data
        if not self.current_day_data: return
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # 2. Get Date string
            if hasattr(self, 'selected_day_label'):
                 date_str = self.selected_day_label
            else:
                 date_str = "Unknown"
            
            # 3. Switch Tab
            self.tabs.setCurrentWidget(self.recorder)
            
            # 4. Load Data into Recorder
            # Recorder expects DataFrame. current_day_data['frames'] is list of dicts.
            frames = self.current_day_data.get('frames', [])
            df = pd.DataFrame(frames)
            
            self.recorder.load_day_data(df, date_str)
        finally:
            QApplication.restoreOverrideCursor()
        
        # Toggle Removed by user request.
        # We now ALWAYS visualize tracking/stow-safe states but we don't switch logic on user toggle.
        pass
             
    def clear(self):
        self.all_data = []
        self.current_day_data = None
        self.current_frames = []
        self.replay_idx = 0
        self.pause()
        self.chart.clear()
        self.table.setRowCount(0)
        self.landscape.update_data([])
        self.lbl_period_theo.setText("0 MWh")
        self.lbl_period_act.setText("0 MWh")
        self.lbl_period_eff.setText("0%")
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0)
        
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # --- TAB 1: DAILY ANALYSIS (Side-by-Side) ---
        self.tab_daily = QWidget()
        self.setup_daily_tab()
        self.tabs.addTab(self.tab_daily, "Daily Analysis")
        
        # --- TAB 2: ANNUAL LANDSCAPE ---
        self.landscape = AnnualLandscapeWidget()
        self.tabs.addTab(self.landscape, "Annual Landscape")
        
        # --- TAB 3: STOW RECORDER ---
        # State will be injected later or taken from parent
        # We need to construct it here but it relies on 'self.state' which is injected after init
        # So we do lazy init or just construct it and injecting state later
        self.recorder = None 
        # We will initialize it in __init__ or after injected -> ResultsWidget init doesn't have state yet
        # Actually Main Window does: self.results.state = self.state
        # So we can create it here if we pass None, but Recorder needs state in init
        # Let's create placeholder and init later? Or just move Results init to accept state?
        # Current MainWindow code: self.results = ResultsWidget(); self.results.state = self.state
        # We'll create it here with a dummy state or wait?
        # Ideally we refactor MainWindow to pass state in constructor.
        # For now, let's just initialize it with None and handle it in setup_ui if state exists??
        # No, setup_ui is called in init.
        # Let's add an explicit 'init_recorder' method called by MainWindow?
        # OR: We modify ResultsWidget to lazily create the tab when state is available.
        
        # Simpler: MainWindow sets attribute `state` then we can use it?
        # No, init runs first.
        # Quick fix: Add `init_tabs` method called after injection?
        # Or just accept state in constructor (best practice but changes interface).
        # Let's just create it but it might crash if state accessed in init.
        # Recorder __init__ does `self.state = state`.
        # If we pass None, it crashes?
        # Let's modify ResultsWidget to NOT create tabs in Init?
        # No, tab_daily is created in setup_ui.
        
        # Let's go with: Create it in `set_state` or property setter.
        
        
    def setup_daily_tab(self):
        layout = QHBoxLayout(self.tab_daily)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # --- LEFT: TABLE & PERIOD STATS ---
        left_widget = QWidget()
        l_layout = QVBoxLayout(left_widget)
        l_layout.setContentsMargins(0,0,0,0)
        
        # Period Totals Header
        self.grp_period = QGroupBox("Simulation Period Totals")
        period_layout = QGridLayout(self.grp_period)
        self.lbl_period_theo = QLabel("0 MWh")
        self.lbl_period_act = QLabel("0 MWh")
        self.lbl_period_eff = QLabel("0%")
        
        period_layout.addWidget(QLabel("Theoretical:"), 0, 0)
        period_layout.addWidget(self.lbl_period_theo, 0, 1)
        period_layout.addWidget(QLabel("Actual:"), 1, 0)
        period_layout.addWidget(self.lbl_period_act, 1, 1)
        period_layout.addWidget(QLabel("Efficiency:"), 2, 0)
        period_layout.addWidget(self.lbl_period_eff, 2, 1)
        
        l_layout.addWidget(self.grp_period)
        
        # Create Stow Profile Button
        self.btn_create_stow = QPushButton("Create Stow Profile for Selected Day")
        self.btn_create_stow.clicked.connect(self.on_create_stow_profile)
        l_layout.addWidget(self.btn_create_stow)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Date", "Theo (kWh)", "Act (kWh)", "Stow %", "Shad %"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self.on_day_selected)
        l_layout.addWidget(self.table)
        
        splitter.addWidget(left_widget)
        
        # --- RIGHT: PLOT & CONTROLS ---
        right_widget = QWidget()
        r_layout = QVBoxLayout(right_widget)
        r_layout.setContentsMargins(10,0,0,0)
        
        # 1. Daily Stats Header
        self.grp_day = QGroupBox("Selected Day Totals")
        day_layout = QHBoxLayout(self.grp_day)
        self.lbl_day_date = QLabel("--")
        self.lbl_day_sun = QLabel("☀ --:-- 🌙 --:--") 
        self.lbl_day_theo = QLabel("0 kWh")
        self.lbl_day_act = QLabel("0 kWh")
        
        day_layout.addWidget(QLabel("Date:"))
        day_layout.addWidget(self.lbl_day_date)
        day_layout.addSpacing(20)
        day_layout.addWidget(self.lbl_day_sun)
        day_layout.addSpacing(20)
        day_layout.addWidget(QLabel("Theoretical:"))
        day_layout.addWidget(self.lbl_day_theo)
        day_layout.addSpacing(20)
        day_layout.addWidget(QLabel("Actual:"))
        day_layout.addWidget(self.lbl_day_act)
        day_layout.addStretch()
        
        r_layout.addWidget(self.grp_day)
        
        # 2. Plot
        # Custom Axis for Time (Hours)
        self.chart = pg.PlotWidget()
        self.chart.setBackground('k')
        self.chart.addLegend()
        self.chart.showGrid(x=True, y=True)
        self.chart.setLabel('left', 'Power (kW)')
        self.chart.setLabel('bottom', 'Time (Hour)')
        self.chart.setMinimumHeight(300)
        
        # Cursor
        self.cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=2))
        self.chart.addItem(self.cursor_line)
        
        r_layout.addWidget(self.chart)
        
        # 3. Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self.on_slider_change)
        r_layout.addWidget(self.slider)
        
        # 4. Play/Pause Controls
        ctrl_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Run")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.clicked.connect(self.pause)
        
        # Step Controls
        self.btn_step_prev = QPushButton("◀ -1")
        self.btn_step_prev.clicked.connect(lambda: self.step_replay(-1))
        self.btn_step_next = QPushButton("+1 ▶")
        self.btn_step_next.clicked.connect(lambda: self.step_replay(1))
        
        self.lbl_time = QLabel("--:--")
        self.lbl_time.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        
        ctrl_row.addWidget(self.btn_play)
        ctrl_row.addWidget(self.btn_pause)
        ctrl_row.addSpacing(20)
        ctrl_row.addWidget(self.btn_step_prev)
        ctrl_row.addWidget(self.lbl_time)
        ctrl_row.addWidget(self.btn_step_next)
        ctrl_row.addStretch()
        
        r_layout.addLayout(ctrl_row)
        
        # 5. Instant Stats Grid
        self.grp_inst = QGroupBox("Timestep Details")
        inst_layout = QGridLayout(self.grp_inst)
        
        self.lbl_az = QLabel("0°")
        self.lbl_el = QLabel("0°")
        self.lbl_dni = QLabel("0 W/m²")
        self.lbl_loss_stow = QLabel("0%")
        self.lbl_loss_shad = QLabel("0%")
        
        inst_layout.addWidget(QLabel("Azimuth:"), 0, 0); inst_layout.addWidget(self.lbl_az, 0, 1)
        inst_layout.addWidget(QLabel("Elevation:"), 0, 2); inst_layout.addWidget(self.lbl_el, 0, 3)
        inst_layout.addWidget(QLabel("DNI:"), 0, 4); inst_layout.addWidget(self.lbl_dni, 0, 5)
        inst_layout.addWidget(QLabel("Stow Loss:"), 1, 0); inst_layout.addWidget(self.lbl_loss_stow, 1, 1)
        inst_layout.addWidget(QLabel("Shadow Loss:"), 1, 2); inst_layout.addWidget(self.lbl_loss_shad, 1, 3)
        
        r_layout.addWidget(self.grp_inst)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 600]) # Initial split 40/60
        
    # --- LOGIC ---
    
    def update_results(self, data):
        if not data: return
        self.all_data = data
        
        # 1. Update Period Stats
        c_theo = sum(d['summary']['theo_kwh'] for d in data)
        c_act  = sum(d['summary']['act_kwh'] for d in data)
        eff = (c_act / c_theo * 100) if c_theo > 0 else 0
        
        self.lbl_period_theo.setText(f"{c_theo/1000:.2f} MWh") # MWh
        self.lbl_period_act.setText(f"{c_act/1000:.2f} MWh")
        self.lbl_period_eff.setText(f"{eff:.1f}%")
        
        # 2. Update Table
        self.table.blockSignals(True)
        self.table.setRowCount(len(data))
        for i, d in enumerate(data):
            s = d['summary']
            t = s['theo_kwh']
            a = s['act_kwh']
            stow_pct = (s['stow_loss_kwh']/t*100) if t>0 else 0
            shad_pct = (s['shad_loss_kwh']/t*100) if t>0 else 0
            
            self.table.setItem(i, 0, QTableWidgetItem(s['date'].strftime("%Y-%m-%d")))
            self.table.setItem(i, 1, QTableWidgetItem(f"{t:.1f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{a:.1f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{stow_pct:.1f}%"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{shad_pct:.1f}%"))
        self.table.blockSignals(False)
        
        # 3. Update Landscape
        self.landscape.update_data(data)
        
        # Select first row
        # Select first row or maintain selection?
        # If we select 0, and it was already 0, the signal won't fire, so current_day_data remains stale.
        # We must force update.
        if len(data) > 0:
             # If selection exists, restore it (if within bounds), otherwise 0
             rows = self.table.selectionModel().selectedRows()
             idx = rows[0].row() if rows else 0
             if idx >= len(data): idx = 0
             
             self.table.selectRow(idx)
             # Force update because data changed underneath
             self.on_day_selected()
             
    def on_day_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows: return
        idx = rows[0].row()
        
        self.current_day_data = self.all_data[idx]
        # Store Date String for Recorder
        s = self.current_day_data['summary'] 
        self.selected_day_label = s['date'].strftime("%Y-%m-%d")
        
        self.load_day_into_plot(self.current_day_data)
        
    def load_day_into_plot(self, day_data):
        self.chart.clear()
        self.chart.addItem(self.cursor_line)
        
        # Frames
        frames = day_data.get('frames', [])
        # Filter Daylight
        self.current_frames = [f for f in frames if f['sun_el'] > 0]
        
        if not self.current_frames: 
             self.lbl_day_sun.setText("No Daylight")
             return
             
        # Sunrise/Sunset
        t_rise = self.current_frames[0]['time'].strftime("%H:%M")
        t_set = self.current_frames[-1]['time'].strftime("%H:%M")
        self.lbl_day_sun.setText(f"☀ {t_rise}  🌙 {t_set}")
        
        # Update Day Header
        s = day_data['summary']
        self.lbl_day_date.setText(s['date'].strftime("%Y-%m-%d"))
        self.lbl_day_theo.setText(f"{s['theo_kwh']:.1f} kWh")
        self.lbl_day_act.setText(f"{s['act_kwh']:.1f} kWh")
        
        # X Axis (Hours of Day)
        # Convert timestamp to hour float
        x = [f['time'].hour + f['time'].minute/60.0 for f in self.current_frames]
        y_theo = [f['theo_w']/1000.0 for f in self.current_frames]
        y_act = [f['act_w']/1000.0 for f in self.current_frames]
        
        self.chart.plot(x, y_theo, pen=pg.mkPen((100,100,100), width=2, style=Qt.PenStyle.DotLine), name="Theoretical")
        self.chart.plot(x, y_act, pen=pg.mkPen('c', width=2), name="Actual")
        
        # Clash/Stow Visualization
        # Identify points where safety=True
        clash_x = []
        clash_y = []
        for i, f in enumerate(self.current_frames):
             if f['safety']:
                  clash_x.append(x[i])
                  clash_y.append(y_act[i])
                  
        if clash_x:
             # Red dots on the actual curve
             scatter = pg.ScatterPlotItem(
                  x=clash_x, y=clash_y, 
                  pen=pg.mkPen(None), brush=pg.mkBrush(255, 50, 50, 255),
                  size=8, symbol='o', name="Clash"
             )
             self.chart.addItem(scatter)
        
        # Configure Slider
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self.current_frames)-1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        
        # Update Instant Stats for frame 0
        self.update_instant_stats(0)
        
        # Reset Playback
        self.replay_idx = 0
        self.pause()
        
        # X-Axis Range
        if x:
            self.chart.setXRange(x[0], x[-1])
            
    def on_slider_change(self, val):
        self.replay_idx = val
        self.update_instant_stats(val)
        
    def step_replay(self, delta):
        # Move slider which triggers update
        val = self.slider.value()
        new_val = max(0, min(self.slider.maximum(), val + delta))
        self.slider.setValue(new_val)

    def _ensure_states(self, f, inactive_override=None, active_override=None):
        """Regenerate physics states if missing OR if override requested."""
        if f.get('states') and inactive_override is None and active_override is None: 
             return
             
        # Lazy Init Kernel
        if not hasattr(self, 'kernel') or self.kernel is None:
             self.kernel = InfiniteKernel()
             
        # Update Kernel Physics
        s = self.state 
        self.kernel.geo = s.geometry
        self.kernel.cfg = s.config
        self.kernel.collider.geo = s.geometry
        self.kernel.collider.cfg = s.config
        
        # Regen Pivots
        pivots = {}
        for r in range(3):
             for c in range(3):
                  y = (r - 1) * s.config.grid_pitch_y
                  x = (c - 1) * s.config.grid_pitch_x
                  pivots[(r,c)] = np.array([float(x), float(y), 0.0])
        self.kernel.pivots = pivots
        
        # Solve
        # Adjust Azimuth for Plant Rotation
        local_az = f['sun_az'] - (-s.config.plant_rotation)
        
        # Use updated solve_timestep signature
        # Convert overrides to local frame if they are global?
        # WAIT. get_position_at returns global values?
        # get_position_at returns raw values stored in Keyframe.
        # Recorder saves global slider values.
        # So yes, they are global. We need to convert to local.
        
        rot = -s.config.plant_rotation
        
        local_in = None
        if inactive_override:
            local_in = (inactive_override[0] - rot, inactive_override[1])
            
        local_act = None
        if active_override:
            local_act = (active_override[0] - rot, active_override[1])

        states, safety = self.kernel.solve_timestep(
             local_az, 
             f['sun_el'], 
             enable_safety=False, # Force Tracking Viz
             inactive_override=local_in,
             active_override=local_act
        )
        
        f['states'] = states
        f['safety'] = safety 
        # print(f"DEBUG: ResultsWidget REGEN STATES. In={local_in} Act={local_act} Safety={safety}") 
        
    def update_instant_stats(self, idx):
        if not self.current_frames or idx >= len(self.current_frames): return
        
        f = self.current_frames[idx]
        self.time_changed.emit(f['time'])
        
        # Check for Profile Override
        in_ovr = None
        act_ovr = None
        
        if hasattr(self, 'state') and self.state.stow_profile:
             pos = self.state.stow_profile.get_position_at(f['time'])
             if pos:
                 # pos is (inactive_az, inactive_el, active_az, active_el)
                 in_ovr = (pos[0], pos[1])
                 act_ovr = (pos[2], pos[3])
                 # print(f"DEBUG: ResultsWidget Profile Active. In={in_ovr} Act={act_ovr}")
             
        # Check/Regen States
        if hasattr(self, 'state'):
             self._ensure_states(f, inactive_override=in_ovr, active_override=act_ovr)
        
        # 1. Update Info Labels
        self.lbl_time.setText(f['time'].strftime("%H:%M"))
        self.lbl_az.setText(f"{f['sun_az']:.1f}°")
        self.lbl_el.setText(f"{f['sun_el']:.1f}°")
        
        # Emit Update
        self.replay_frame.emit(f['sun_az'], f['sun_el'], f['safety'], f['states'])
        
        # Calc Losses %
        t = f.get('theo_w', 0)
        a = f.get('act_w', 0)
        stow_w = f.get('stow_w', 0)
        shad_w = f.get('shad_w', 0)
        dni = f.get('dni', 0)
        
        stow_pct = (stow_w / t * 100) if t > 0 else 0
        shad_pct = (shad_w / t * 100) if t > 0 else 0
        
        self.lbl_loss_stow.setText(f"{stow_pct:.1f}%")
        self.lbl_loss_shad.setText(f"{shad_pct:.1f}%")
             
        # DNI
        self.lbl_dni.setText(f"{dni:.1f} W/m²") 
        
        # 2. Update Cursor Line
        x_val = f['time'].hour + f['time'].minute/60.0
        self.cursor_line.setValue(x_val)
        
        # 3. Emit Signal for 3D View
        self.replay_frame.emit(f['sun_az'], f['sun_el'], f['safety'], f.get('states', []))

    def toggle_play(self):
        if self.is_playing:
             self.pause()
        else:
             self.play()
             
    def play(self):
        # Loop Check
        if self.replay_idx >= len(self.current_frames) - 1:
            self.replay_idx = 0
            self.slider.setValue(0)
            
        self.is_playing = True
        self.btn_play.setText("Playing...")
        self.btn_play.setStyleSheet("background-color: #4CAF50; color: white;")
        self.timer.start()
        
    def pause(self):
        self.is_playing = False
        self.btn_play.setText("▶ Run")
        self.btn_play.setStyleSheet("")
        self.timer.stop()
        
    def next_frame(self):
        if not self.is_playing: return
        
        self.replay_idx += 1
        if self.replay_idx >= len(self.current_frames):
             self.pause()
             return
             
        self.slider.blockSignals(True)
        self.slider.setValue(self.replay_idx)
        self.slider.blockSignals(False)
        self.update_instant_stats(self.replay_idx)
