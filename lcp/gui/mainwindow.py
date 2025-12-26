
from PyQt6.QtWidgets import (QMainWindow, QDockWidget, QWidget)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt

from lcp.gui.state import AppState
from lcp.gui.viewport import PlantViewport
from lcp.gui.sidebar import Sidebar
from lcp.gui.results_widget import ResultsWidget
from lcp.gui.recorder import StowRecorder
from lcp.gui.runner import SimulationWorker

from lcp.core.geometry import PanelGeometry
from lcp.core.config import ScenarioConfig
from lcp.physics.engine import InfiniteKernel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LCP-Sim Desktop v1.0")
        self.resize(1280, 800)
        self.showMaximized()
        
        # Icon
        import os
        icon_path = os.path.join(os.path.dirname(__file__), 'app_icon.png')
        if os.path.exists(icon_path):
             self.setWindowIcon(QIcon(icon_path))
        
        # 1. Init State (Defaults)
        geo = PanelGeometry(width=1.46, length=1.46, thickness=0.15, pivot_offset=(0,0,0.38-0.075))
        cfg = ScenarioConfig(grid_pitch_x=1.7, grid_pitch_y=1.7, total_panels=240)
        self.state = AppState(geometry=geo, config=cfg)
        
        # 2. Central 3D View
        self.viewport = PlantViewport(self.state, self)
        self.setCentralWidget(self.viewport)
        
        # 3. Create Docks
        self.create_docks()
        
        # 4. Status Bar
        self.statusBar().showMessage("Ready")
        
    def create_docks(self):
        # A. LEFT: Control Room (Sidebar)
        dock_left = QDockWidget("Control Room", self)
        dock_left.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self.sidebar = Sidebar(self.state)
        dock_left.setWidget(self.sidebar)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_left)
        
        # B. RIGHT: Results
        dock_right = QDockWidget("Simulation Results", self)
        dock_right.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.results = ResultsWidget()
        self.results.state = self.state # Inject AppState for shadow reconstruction
        dock_right.setWidget(self.results)
        self.results.viewport = self.viewport # Inject Viewport for Recorder
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_right)
        
        # C. BOTTOM: Stow Recorder
        # (REMOVED - integrated into ResultsWidget)
        
        # --- CONNECTIONS ---
        
        # 0. Visual Settings
        self.sidebar.cb_show_pivots.toggled.connect(self.viewport.set_show_pivots)
        self.sidebar.cb_show_rays.toggled.connect(self.viewport.set_show_rays)
        self.sidebar.cb_show_full.toggled.connect(self.viewport.set_show_full_plant)
        # self.sidebar.cb_stow.toggled.connect(self.results.set_enable_safety) # Toggle Removed
        self.sidebar.cb_show_tolerance.toggled.connect(self.viewport.set_show_tolerance)
        
        # 1. Sidebar -> Viewport (Live Param Update)
        self.sidebar.geometry_changed.connect(self.viewport.update_scene)
        
        # Connect Simulation Runner
        self.sidebar.run_requested.connect(self.start_simulation)
        self.sidebar.save_requested.connect(self.on_save_requested)
        
        # Connect Replay Viz
        self.results.replay_frame.connect(self.viewport.update_from_frame)
        
        self.sidebar.load_requested.connect(self.on_load_finished)
        
        # 3. Recorder -> Viewport (Animation/Manual)
        # Note: Recorder is now inside ResultsWidget. We need to expose its signal.
        # This will be wired up after ResultsWidget is fully initialized or via a new signal on ResultsWidget proxy
        
        # 4. Init Teach Kernel (For Manual Mode Physics)
        self.teach_kernel = InfiniteKernel(self.state.geometry, self.state.config)
        self.sidebar.geometry_changed.connect(self.update_teach_kernel)
        
    def update_teach_kernel(self):
        # Re-init kernel on geometry change
        self.teach_kernel = InfiniteKernel(self.state.geometry, self.state.config)
        
    def on_manual_override(self, sun_az, sun_el, safety, states):
        # Recorder now calculates physics internally and just asks for visualization
        # so this method simplifies to just pass-through to viewport
        self.viewport.update_from_frame(sun_az, sun_el, safety, states)

    # Legacy method signature was (az, el). Keeping for reference but unused mainly:
    def _legacy_manual_override(self, az, el):
        # Use Teach Kernel to solve state with override
        # This provides collision detection ("clash profile") and parity mixing
        dt = self.recorder.current_time # FAIL: Recorder no longer member of Main
        sun = self.recorder.solar.get_position(dt)
        local_az = sun.azimuth - self.state.config.plant_rotation
        
        # Solve
        print(f"DEBUG: Manual Override Az={az} El={el} Time={dt}")
        states, safety = self.teach_kernel.solve_timestep(
             local_az, sun.elevation, 
             enable_safety=True, 
             stow_override=(az, el)
        )
        
        # Update Viewport
        # Update Viewport
        # print(f"DEBUG: Updating Viewport from Manual Override")
        self.viewport.update_from_frame(sun_az, sun_el, safety, states)
        
    def start_simulation(self):
        # Disable Run Button
        self.sidebar.set_running()
        self.statusBar().showMessage("Starting Simulation...")
        
        # Update Teach Kernel to match latest config
        self.update_teach_kernel()
        
        # Settings from State
        start = self.state.sim_settings.start_date
        days = 365 if self.state.sim_settings.full_year else self.state.sim_settings.duration_days
        
        # Worker
        import datetime
        dt_start = datetime.datetime.combine(start, datetime.time.min)
        
        self.worker = SimulationWorker(self.state, dt_start, days)
        # Signals
        self.worker.progress.connect(self.on_progress)
        self.worker.status.connect(self.statusBar().showMessage)
        self.worker.finished_data.connect(self.on_sim_finished)
        self.worker.error.connect(lambda e: self.statusBar().showMessage(f"Error: {e}"))
        
        self.worker.start()
        
    def on_progress(self, val):
        self.sidebar.btn_run.setText(f"Running... {val}%")
        
    def on_sim_finished(self, data):
        self.sidebar.reset_ready()
        self.results.update_results(data)
        self.statusBar().showMessage(f"Simulation Complete. {len(data)} Days Processed.")
        
    def on_load_finished(self, data):
        self.results.update_results(data)
        self.statusBar().showMessage(f"Loaded {len(data)} Days.")
        
    def on_save_requested(self, name):
         self.statusBar().showMessage(f"Saving to {name}...")
         data = self.results.all_data
         if not data:
              self.statusBar().showMessage("No data to save.")
              return
         try:
              # Ensure PM base path is updated incase sidebar changed it
              # (It shares the state.storage_path but PM object in Sidebar might need update if we used a new instance)
              # But Sidebar.pm is updated in update_path_manual.
              clean = self.sidebar.pm.save_simulation(name, self.state.geometry, self.state.config, data)
              self.sidebar.refresh_load_list()
              self.statusBar().showMessage(f"Saved: {clean}")
         except Exception as e:
              self.statusBar().showMessage(f"Save Failed: {e}")
