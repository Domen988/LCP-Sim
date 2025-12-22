
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QDoubleSpinBox, QSpinBox, QCheckBox, QDateEdit, 
                             QGroupBox, QComboBox, QFormLayout, QFileDialog)
from PyQt6.QtCore import pyqtSignal, QDate
from datetime import date

from lcp.gui.state import AppState
from lcp.core.persistence import PersistenceManager

class Sidebar(QWidget):
    """
    Control Room Sidebar.
    Replicates the Streamlit sidebar sections:
    1. Persistence (Save/Load)
    2. Geometry Config
    3. Plant Sizing
    4. Simulation Settings
    """
    
    # Signals to notify parent to refresh Viewport or Run Sim
    geometry_changed = pyqtSignal()
    run_requested = pyqtSignal()
    load_requested = pyqtSignal(list) # Emits results data
    
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.pm = PersistenceManager(base_path=state.storage_path)
        
        self.init_persistence()
        self.init_geometry()
        self.init_sizing()
        self.init_sim_settings()
        
        self.layout.addStretch()
        
    def init_persistence(self):
        gb = QGroupBox("Simulation Management")
        form = QFormLayout()
        
        # Path
        self.path_edit = QLineEdit(self.state.storage_path)
        self.path_edit.editingFinished.connect(self.update_path)
        form.addRow("Path:", self.path_edit)
        
        # Save
        self.save_name = QLineEdit("New_Sim")
        btn_save = QPushButton("Save Current")
        btn_save.clicked.connect(self.save_current)
        form.addRow(self.save_name, btn_save)
        
        # Load
        self.combo_load = QComboBox()
        self.refresh_load_list()
        btn_load = QPushButton("Load")
        btn_load.clicked.connect(self.load_sim)
        form.addRow(self.combo_load, btn_load)
        
        gb.setLayout(form)
        self.layout.addWidget(gb)
        
    def update_path(self):
        new_path = self.path_edit.text()
        self.state.storage_path = new_path
        self.pm.base_path = new_path
        self.refresh_load_list()

    def refresh_load_list(self):
        self.combo_load.clear()
        try:
             sims = self.pm.list_simulations()
             self.combo_load.addItems(sims)
        except:
             pass

    def save_current(self):
        # TODO: Access results from MainWindow? 
        # For now, we rely on parent to handle actual save or pass results here.
        # Ideally, MainWindow handles saving.
        pass

    def load_sim(self):
        name = self.combo_load.currentText()
        if not name: return
        try:
            geo, cfg, res = self.pm.load_simulation(name)
            # Update State
            self.state.geometry = geo
            self.state.config = cfg
            # Update Widgets
            self.update_widgets_from_state()
            # Notify
            self.geometry_changed.emit()
            self.load_requested.emit(res)
        except Exception as e:
            print(f"Load Error: {e}")

    def init_geometry(self):
        gb = QGroupBox("Geometry Config")
        form = QFormLayout()
        
        def add_float(label, attr, min_v, max_v, step, obj):
            sb = QDoubleSpinBox()
            sb.setRange(min_v, max_v)
            sb.setSingleStep(step)
            sb.setValue(getattr(obj, attr))
            sb.valueChanged.connect(lambda v: self.on_geo_change(obj, attr, v))
            form.addRow(label, sb)
            self.__setattr__(f"sb_{attr}", sb) # Keep ref
            
        g = self.state.geometry
        c = self.state.config
        
        add_float("Width (m)", "width", 0.5, 5.0, 0.1, g)
        add_float("Length (m)", "length", 0.5, 5.0, 0.1, g)
        add_float("Pivot Depth (m)", "pivot_depth_glass", 0.0, 0.5, 0.01, self) # Proxy
        # Real geo uses pivot_offset vector. Proxy needed.
        # Pivot offset z = pivot_depth - thickness/2
        
        add_float("Thickness (m)", "thickness", 0.01, 0.5, 0.01, g)
        add_float("Pitch X (m)", "grid_pitch_x", 0.5, 10.0, 0.1, c)
        add_float("Pitch Y (m)", "grid_pitch_y", 0.5, 10.0, 0.1, c)
        
        gb.setLayout(form)
        self.layout.addWidget(gb)
        
    # Pivot Depth Proxy handling
    @property
    def pivot_depth_glass(self):
        # off_z = depth - t/2
        # depth = off_z + t/2
        return self.state.geometry.pivot_offset[2] + (self.state.geometry.thickness / 2)
        
    @pivot_depth_glass.setter
    def pivot_depth_glass(self, val):
        t = self.state.geometry.thickness
        self.state.geometry.pivot_offset = (0, 0, val - t/2)

    def on_geo_change(self, obj, attr, val):
        if obj == self: # Proxy
             setattr(self, attr, val)
        else:
             setattr(obj, attr, val)
        self.geometry_changed.emit()

    def init_sizing(self):
        gb = QGroupBox("Plant Sizing")
        form = QFormLayout()
        
        self.sb_rows = QSpinBox()
        self.sb_rows.setRange(1, 1000)
        self.sb_rows.setValue(self.state.rows)
        self.sb_rows.valueChanged.connect(self.on_size_change)
        
        self.sb_cols = QSpinBox()
        self.sb_cols.setRange(1, 1000)
        self.sb_cols.setValue(self.state.cols)
        self.sb_cols.valueChanged.connect(self.on_size_change)
        
        self.lbl_total = QLabel(f"{self.state.rows * self.state.cols}")
        
        form.addRow("Rows", self.sb_rows)
        form.addRow("Cols", self.sb_cols)
        form.addRow("Total Panels:", self.lbl_total)
        
        gb.setLayout(form)
        self.layout.addWidget(gb)
        
    def on_size_change(self):
        self.state.rows = self.sb_rows.value()
        self.state.cols = self.sb_cols.value()
        self.state.config.total_panels = self.state.rows * self.state.cols
        self.lbl_total.setText(str(self.state.config.total_panels))
        self.geometry_changed.emit()

    def init_sim_settings(self):
        gb = QGroupBox("Simulation Settings")
        form = QFormLayout()
        
        s = self.state.sim_settings
        
        self.de_start = QDateEdit()
        self.de_start.setDate(QDate(s.start_date.year, s.start_date.month, s.start_date.day))
        self.de_start.dateChanged.connect(self.on_sim_change)
        form.addRow("Start Date", self.de_start)
        
        self.cb_full = QCheckBox("Full Year")
        self.cb_full.setChecked(s.full_year)
        self.cb_full.toggled.connect(self.on_sim_change)
        form.addRow(self.cb_full)
        
        self.sb_days = QSpinBox()
        self.sb_days.setRange(1, 3650)
        self.sb_days.setValue(s.duration_days)
        self.sb_days.valueChanged.connect(self.on_sim_change)
        form.addRow("Duration (Days)", self.sb_days)
        
        self.sb_step = QSpinBox()
        self.sb_step.setRange(1, 60)
        self.sb_step.setValue(s.timestep_min)
        self.sb_step.valueChanged.connect(self.on_sim_change)
        form.addRow("Timestep (min)", self.sb_step)
        
        self.btn_run = QPushButton("▶ RUN SIMULATION")
        self.btn_run.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 5px;")
        self.btn_run.clicked.connect(self.run_requested.emit)
        
        gb.setLayout(form)
        self.layout.addWidget(gb)
        self.layout.addWidget(self.btn_run)
        
    def on_sim_change(self):
        s = self.state.sim_settings
        qdate = self.de_start.date()
        s.start_date = date(qdate.year(), qdate.month(), qdate.day())
        s.full_year = self.cb_full.isChecked()
        s.duration_days = self.sb_days.value()
        s.timestep_min = self.sb_step.value()
        
        self.sb_days.setEnabled(not s.full_year)
        
    def update_widgets_from_state(self):
        # Block signals to prevent feedback loops?
        # Ideally yes. For now, direct set.
        pass
