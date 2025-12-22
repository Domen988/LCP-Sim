
from PyQt6.QtWidgets import (QMainWindow, QDockWidget, QWidget)
from PyQt6.QtCore import Qt

from lcp.gui.state import AppState
from lcp.gui.viewport import PlantViewport
from lcp.gui.sidebar import Sidebar
from lcp.gui.results_widget import ResultsWidget
from lcp.gui.recorder import StowRecorder
from lcp.gui.runner import SimulationWorker

from lcp.core.geometry import PanelGeometry
from lcp.core.config import ScenarioConfig

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LCP-Sim Desktop v1.0")
        self.resize(1600, 900)
        
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
        dock_right.setWidget(self.results)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_right)
        
        # C. BOTTOM: Stow Recorder
        dock_bot = QDockWidget("Stow Recorder", self)
        dock_bot.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.recorder = StowRecorder(self.state)
        dock_bot.setWidget(self.recorder)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock_bot)
        
        # --- CONNECTIONS ---
        
        # 1. Sidebar -> Viewport (Live Param Update)
        self.sidebar.geometry_changed.connect(self.viewport.update_scene)
        
        # Connect Simulation Runner
        self.sidebar.run_requested.connect(self.start_simulation)
        
        # Connect Replay Viz
        self.results.replay_frame.connect(self.viewport.update_from_frame)
        
        self.sidebar.load_requested.connect(self.on_load_finished)
        
        # 3. Recorder -> Viewport (Animation/Manual)
        self.recorder.scene_update.connect(self.viewport.update_scene)
        
    def start_simulation(self):
        # Disable Run Button
        self.sidebar.btn_run.setEnabled(False)
        self.statusBar().showMessage("Starting Simulation...")
        
        # Settings from State
        start = self.state.sim_settings.start_date
        days = 365 if self.state.sim_settings.full_year else self.state.sim_settings.duration_days
        
        # Worker
        import datetime
        dt_start = datetime.datetime.combine(start, datetime.time.min)
        
        self.worker = SimulationWorker(self.state, dt_start, days)
        # Signals
        self.worker.progress.connect(self.on_progress) # Need to add ProgressBar to StatusBar or Sidebar?
        self.worker.status.connect(self.statusBar().showMessage)
        self.worker.finished_data.connect(self.on_sim_finished)
        self.worker.error.connect(lambda e: self.statusBar().showMessage(f"Error: {e}"))
        
        self.worker.start()
        
    def on_progress(self, val):
        # Update progress bar on Sidebar button? Or Status Bar
        self.sidebar.btn_run.setText(f"Running... {val}%")
        
    def on_sim_finished(self, data):
        self.sidebar.btn_run.setEnabled(True)
        self.sidebar.btn_run.setText("▶ RUN SIMULATION")
        self.results.update_results(data)
        self.statusBar().showMessage(f"Simulation Complete. {len(data)} Days Processed.")
        
    def on_load_finished(self, data):
        self.results.update_results(data)
        self.statusBar().showMessage(f"Loaded {len(data)} Days.")
