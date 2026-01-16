
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QDoubleSpinBox, QSpinBox, QCheckBox, QDateEdit, 
                             QGroupBox, QComboBox, QFormLayout, QFileDialog, QToolButton, 
                             QScrollArea, QSizePolicy, QMessageBox, QHBoxLayout, QApplication)
from PyQt6.QtCore import pyqtSignal, QDate, Qt, QPropertyAnimation, QParallelAnimationGroup
from datetime import date

from lcp.gui.state import AppState
from lcp.core.persistence import PersistenceManager
from lcp.gui.load_dialog import SimulationLoadDialog
from PyQt6.QtWidgets import QStyle

class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None, expanded=False):
        super().__init__(parent)
        self.toggle_button = QToolButton(text=title, checkable=True, checked=expanded)
        self.toggle_button.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.toggle_animation = QParallelAnimationGroup(self)
        self.content_area = QScrollArea(maximumHeight=0, minimumHeight=0)
        self.content_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.content_area.setFrameShape(QScrollArea.Shape.NoFrame)

        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.animation = QPropertyAnimation(self.content_area, b"maximumHeight")
        self.animation.setDuration(300)
        self.toggle_animation.addAnimation(self.animation)
        
        self.content_layout = QVBoxLayout()
        self.content_widget = QWidget()
        self.content_widget.setLayout(self.content_layout)
        self.content_area.setWidget(self.content_widget)
        self.content_area.setWidgetResizable(True)
        
        if expanded:
             self.content_area.setMaximumHeight(1000) # Arbitrary large
             self.content_area.setMinimumHeight(0)

    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if not checked else Qt.ArrowType.RightArrow)
        self.toggle_animation.setDirection(QPropertyAnimation.Direction.Forward if not checked else QPropertyAnimation.Direction.Backward)
        
        # Calculate height
        content_height = self.content_widget.sizeHint().height()
        
        self.animation.setStartValue(0)
        self.animation.setEndValue(content_height)
        self.toggle_animation.start()

    def setContentLayout(self, layout):
        pass
        
    def addLayout(self, layout):
        self.content_layout.addLayout(layout)
        
    def addWidget(self, widget):
        self.content_layout.addWidget(widget)

class Sidebar(QWidget):
    """
    Control Room Sidebar.
    Replicates the Streamlit sidebar sections.
    """
    
    # Signals
    geometry_changed = pyqtSignal()
    run_requested = pyqtSignal()
    load_requested = pyqtSignal(list) # Emits results data
    save_requested = pyqtSignal(str) # Emits name to save
    stow_all_changed = pyqtSignal(bool) # Emits stow all state
    
    # Styles
    STYLE_READY = "background-color: #2e7d32; color: white; font-weight: bold; padding: 5px;"
    STYLE_STALE = "background-color: #ef6c00; color: white; font-weight: bold; padding: 5px;"
    STYLE_RUNNING = "background-color: #555555; color: white; padding: 5px;"

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
        self.init_visual_settings()
        
        self.layout.addStretch()
        
    def init_persistence(self):
        # Default Collapsed
        box = CollapsibleBox("Simulation Management", expanded=False)
        form = QFormLayout()
        
        # Path
        path_layout = QHBoxLayout() # Changed to HBox for inline
        self.path_edit = QLineEdit(self.state.storage_path)
        self.path_edit.editingFinished.connect(self.update_path_manual)
        
        btn_browse = QPushButton()
        btn_browse.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
        btn_browse.setToolTip("Browse Folder")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(self.browse_path)
        
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(btn_browse)
        form.addRow("Path:", path_layout)
        
        # Save
        self.save_name = QLineEdit("New_Sim")
        btn_save = QPushButton("Save Current")
        btn_save.clicked.connect(self.save_current)
        form.addRow(self.save_name, btn_save)
        
        # Load / Delete
        # Replaced Combo with Modal logic
        self.lbl_current_sim = QLabel("No Simulation Loaded")
        self.lbl_current_sim.setStyleSheet("font-weight: bold; color: #444;")
        form.addRow("Loaded:", self.lbl_current_sim)
        
        btn_layout = QHBoxLayout()
        btn_load = QPushButton("Load Simulation...")
        btn_load.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        btn_load.clicked.connect(self.open_load_dialog)
        
        btn_del = QPushButton("Delete Loaded")
        btn_del.setStyleSheet("color: red;")
        btn_del.clicked.connect(self.delete_sim)
        
        btn_layout.addWidget(btn_load)
        btn_layout.addWidget(btn_del)
        
        form.addRow(btn_layout)
        
        form.addRow(btn_layout)
        
        # Export Stow Button (Moved here per user request)
        self.btn_export_stow = QPushButton("Save stow strategy to csv")
        self.btn_export_stow.clicked.connect(self.export_stow_strategy)
        self.btn_export_stow.setEnabled(False) # Default disabled until load
        form.addRow(self.btn_export_stow)
        
        box.addLayout(form)
        self.layout.addWidget(box)
        
    def browse_path(self):
        d = QFileDialog.getExistingDirectory(self, "Select Simulation Data Folder", self.state.storage_path)
        if d:
            self.path_edit.setText(d)
            self.update_path_manual()
            
    def update_path_manual(self):
        new_path = self.path_edit.text()
        self.state.storage_path = new_path
        self.pm.base_path = new_path
        self.refresh_load_list()

    def refresh_load_list(self):
        # Only refresh stow generator combo if it exists
        if hasattr(self, 'cmb_stg_source'):
            self.cmb_stg_source.clear()
            try:
                 sims = self.pm.list_simulations()
                 self.cmb_stg_source.addItems(sims)
            except: pass

    def open_load_dialog(self):
        sims = self.pm.list_simulations_details()
        dlg = SimulationLoadDialog(sims, self)
        if dlg.exec():
            name = dlg.selected_sim
            if name:
                self.do_load_sim(name)
                
    def do_load_sim(self, name):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            geo, cfg, res = self.pm.load_simulation(name)
            
            self.state.geometry = geo
            self.state.config = cfg
            
            self.lbl_current_sim.setText(name)
            
            # Update all GUI elements
            self.update_widgets_from_state()
            self.geometry_changed.emit()
            self.load_requested.emit(res)
            self.reset_ready()
            
            if hasattr(self, 'btn_export_stow'):
                self.btn_export_stow.setEnabled(True)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Load Error", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def load_sim(self):
        # Legacy stub or redirect if needed
        pass

    def save_current(self):
        name = self.save_name.text()
        if not name: return
        
        # Check Overwrite (Case-sensitive check)
        existing = [s for s in self.pm.list_simulations() if s == name]
        if existing:
             ret = QMessageBox.question(self, "Overwrite Simulation", 
                                        f"Simulation '{name}' already exists.\nOverwrite it?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
             if ret == QMessageBox.StandardButton.No:
                  return
                  
        self.save_requested.emit(name)
        
    def delete_sim(self):
        name = self.lbl_current_sim.text()
        if not name or name == "No Simulation Loaded": return
        
        ret = QMessageBox.warning(self, "Delete Simulation",
                                  f"Are you sure you want to PERMANENTLY delete '{name}'?\nThis cannot be undone.",
                                  QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ret == QMessageBox.StandardButton.Yes:
             success = self.pm.delete_simulation(name)
             if success:
                  self.refresh_load_list()
                  # "Close" the simulation (Clear results)
                  self.load_requested.emit([])
             else:
                  QMessageBox.critical(self, "Error", "Failed to delete simulation.")

    def load_sim(self):
        name = self.combo_load.currentText()
        if not name: return
        try:
            geo, cfg, res = self.pm.load_simulation(name)
            self.state.geometry = geo
            self.state.config = cfg
            self.update_widgets_from_state()
            self.geometry_changed.emit()
            self.load_requested.emit(res)
            self.geometry_changed.emit()
            self.load_requested.emit(res)
            # Parameters match results now
            self.reset_ready()
            
            # Enable Export if button exists
            if hasattr(self, 'btn_export_stow'):
                self.btn_export_stow.setEnabled(True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Load Error", f"Failed to load simulation:\n{e}")
            print(f"Load Error: {e}")

    def on_geo_change(self, category, attr, val):
        if category == 'self':
             setattr(self, attr, val)
        elif category == 'geo':
             setattr(self.state.geometry, attr, val)
        elif category == 'conf':
             setattr(self.state.config, attr, val)
             
        self.geometry_changed.emit()
        self.set_stale()

    def init_geometry(self):
        box = CollapsibleBox("Geometry Config", expanded=False)
        form = QFormLayout()
        
        def add_float(label, attr, min_v, max_v, step, category):
            sb = QDoubleSpinBox()
            sb.setRange(min_v, max_v)
            sb.setSingleStep(step)
            
            # Resolve initial value (Just for init)
            if category == 'geo': obj = self.state.geometry
            elif category == 'conf': obj = self.state.config
            else: obj = self
            
            sb.setValue(getattr(obj, attr))
            sb.valueChanged.connect(lambda v: self.on_geo_change(category, attr, v))
            form.addRow(label, sb)
            self.__setattr__(f"sb_{attr}", sb) 
            
        add_float("Width (m)", "width", 0.5, 5.0, 0.1, 'geo')
        add_float("Length (m)", "length", 0.5, 5.0, 0.1, 'geo')
        add_float("Pivot Depth (m)", "pivot_depth_glass", 0.0, 0.5, 0.01, 'self')
        add_float("Thickness (m)", "connected_thickness", 0.01, 0.5, 0.01, 'self')
        add_float("Pitch X (m)", "grid_pitch_x", 0.5, 10.0, 0.1, 'conf')
        add_float("Pitch Y (m)", "grid_pitch_y", 0.5, 10.0, 0.1, 'conf')
        add_float("Field Spacing X (m)", "field_spacing_x", 0.5, 10.0, 0.1, 'conf')
        add_float("Field Spacing Y (m)", "field_spacing_y", 0.5, 10.0, 0.1, 'conf')
        add_float("Clash Tolerance (m)", "tolerance", 0.0, 0.5, 0.01, 'conf')
        add_float("Min Elevation (°)", "min_elevation", 0.0, 90.0, 1.0, 'conf')
        
        box.addLayout(form)
        self.layout.addWidget(box)

    @property
    def pivot_depth_glass(self):
        return self.state.geometry.pivot_offset[2] + (self.state.geometry.thickness / 2)
        
    @pivot_depth_glass.setter
    def pivot_depth_glass(self, val):
        t = self.state.geometry.thickness
        self.state.geometry.pivot_offset = (0, 0, val - t/2)
        
    @property
    def connected_thickness(self):
        return self.state.geometry.thickness
        
    @connected_thickness.setter
    def connected_thickness(self, new_thickness):
        # Invariant: Pivot Depth (Glass to Pivot)
        current_depth = self.pivot_depth_glass
        
        # Update Thickness
        self.state.geometry.thickness = new_thickness
        
        # Restore Depth by moving Offset
        # Depth = Offset + T/2  =>  Offset = Depth - T/2
        self.state.geometry.pivot_offset = (0, 0, current_depth - new_thickness/2)

    def init_sizing(self):
        box = CollapsibleBox("Plant Config", expanded=False)
        form = QFormLayout()
        
        # Sizing
        self.sb_rows = QSpinBox()
        self.sb_rows.setRange(1, 1000)
        self.sb_rows.setValue(self.state.rows)
        self.sb_rows.valueChanged.connect(self.on_size_change)
        
        self.sb_cols = QSpinBox()
        self.sb_cols.setRange(1, 1000)
        self.sb_cols.setValue(self.state.cols)
        self.sb_cols.valueChanged.connect(self.on_size_change)
        
        self.lbl_total = QLabel(f"{self.state.rows * self.state.cols}")
        
        # Rotation
        self.sb_rotation = QDoubleSpinBox()
        self.sb_rotation.setRange(-180.0, 180.0)
        self.sb_rotation.setSingleStep(1.0)
        self.sb_rotation.setValue(self.state.config.plant_rotation)
        self.sb_rotation.valueChanged.connect(self.on_rotation_change)

        form.addRow("Rows", self.sb_rows)
        form.addRow("Cols", self.sb_cols)
        form.addRow("Total Panels:", self.lbl_total)
        form.addRow("Plant Rotation (°)", self.sb_rotation)
        
        box.addLayout(form)
        self.layout.addWidget(box)
        
    def on_size_change(self):
        self.state.rows = self.sb_rows.value()
        self.state.cols = self.sb_cols.value()
        self.state.config.total_panels = self.state.rows * self.state.cols
        self.lbl_total.setText(str(self.state.config.total_panels))
        self.geometry_changed.emit()
        self.set_stale()
        
    def on_rotation_change(self):
        self.state.config.plant_rotation = self.sb_rotation.value()
        self.geometry_changed.emit() # Triggers kernel regen in MainWindow
        self.set_stale()

    def init_sim_settings(self):
        box = CollapsibleBox("Simulation Settings", expanded=True) 
        # override: collapsed
        box.toggle_button.setChecked(False) 
        
        form = QFormLayout()
        s = self.state.sim_settings
        
        self.de_start = QDateEdit()
        self.de_start.setCalendarPopup(True)
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
        self.sb_step.setValue(s.timestep_min)
        self.sb_step.setValue(s.timestep_min)
        self.sb_step.valueChanged.connect(self.on_sim_change)
        form.addRow("Timestep (min)", self.sb_step)
        
        # Sun Source
        self.cmb_sun_source = QComboBox()
        self.cmb_sun_source.addItems(["pvlib", "csv"])
        self.cmb_sun_source.setCurrentText(self.state.config.sun_source)
        self.cmb_sun_source.currentTextChanged.connect(self.on_sim_change)
        form.addRow("Sun Data Source", self.cmb_sun_source)
        
        # Clashes Only Settings
        self.cb_clashes = QCheckBox("Run Clashes Only (Dec 21)")
        self.cb_clashes.setChecked(s.clashes_only_mode)
        self.cb_clashes.toggled.connect(self.on_sim_change)
        form.addRow(self.cb_clashes)
        
        self.sb_window = QSpinBox()
        self.sb_window.setRange(1, 100)
        self.sb_window.setValue(s.clash_window)
        self.sb_window.valueChanged.connect(self.on_sim_change)
        form.addRow("Days +/- Solstice", self.sb_window)
        
        self.sb_min_el = QDoubleSpinBox()
        self.sb_min_el.setRange(0, 90)
        self.sb_min_el.setValue(s.clash_min_elevation)
        self.sb_min_el.valueChanged.connect(self.on_sim_change)
        form.addRow("Min Elevation (°)", self.sb_min_el)
        
        box.addLayout(form)
        self.layout.addWidget(box)
        
        # Stow Strategy Generator Section
        self.init_stow_gen()
        
        # Safe Contour Mapping Section
        self.init_contour_map()
        
        # Run Button outside
        self.btn_run = QPushButton("▶ RUN SIMULATION")
        self.btn_run.setStyleSheet(self.STYLE_READY)
        self.btn_run.clicked.connect(self.run_requested.emit)
        self.layout.addWidget(self.btn_run)

    def init_stow_gen(self):
        box = CollapsibleBox("Stow Strategy Generator", expanded=False)
        form = QFormLayout()

        # Inputs
        self.cmb_stg_source = QComboBox()
        try:
            self.cmb_stg_source.addItems(self.pm.list_simulations())
        except: pass
        
        # Strategy Mode
        self.cmb_stg_mode = QComboBox()
        self.cmb_stg_mode.addItems(["Counter-Rotation", "Clash Contour"])
        self.cmb_stg_mode.currentTextChanged.connect(self.update_stow_ui_visibility)
        
        # Clash Contour Map Selection
        self.cmb_stg_contour = QComboBox()
        self.refresh_contour_list() # Initial populate
        
        # Params
        self.sb_stg_min_gap = QSpinBox() # For Counter-Rotation mostly
        self.sb_stg_min_gap.setRange(1, 60); self.sb_stg_min_gap.setValue(5)
        
        self.sb_stg_safe_el = QDoubleSpinBox()
        self.sb_stg_safe_el.setRange(0, 90); self.sb_stg_safe_el.setValue(30.0)
        
        self.sb_stg_offset = QDoubleSpinBox()
        self.sb_stg_offset.setRange(0, 180); self.sb_stg_offset.setValue(45.0)
        
        self.sb_stg_speed = QDoubleSpinBox()
        self.sb_stg_speed.setRange(0.1, 120); self.sb_stg_speed.setValue(5.0) # Default changed to 5 as requested
        
        self.sb_stg_buffer = QDoubleSpinBox()
        self.sb_stg_buffer.setRange(0.0, 10.0); self.sb_stg_buffer.setValue(0.5); self.sb_stg_buffer.setSingleStep(0.1)
        
        # Add Rows
        form.addRow("Source Simulation:", self.cmb_stg_source)
        form.addRow("Strategy Mode:", self.cmb_stg_mode)
        
        # Dynamic Rows (Labels referenced for hiding?)
        # For simplicity, we add all and toggle visibility of widgets + labels
        self.lbl_contour = QLabel("Clash Contour Map:")
        form.addRow(self.lbl_contour, self.cmb_stg_contour)
        
        self.lbl_min_gap = QLabel("Min Safe Interval (min):")
        form.addRow(self.lbl_min_gap, self.sb_stg_min_gap)
        
        self.lbl_safe_el = QLabel("Safe Stow El (°):")
        form.addRow(self.lbl_safe_el, self.sb_stg_safe_el)
        
        self.lbl_offset = QLabel("Westward Offset (°):")
        form.addRow(self.lbl_offset, self.sb_stg_offset)
        
        form.addRow("Elevation Motor Speed (°/min):", self.sb_stg_speed)
        
        self.lbl_buffer = QLabel("Elevation Buffer (°):")
        form.addRow(self.lbl_buffer, self.sb_stg_buffer)
        
        btn_gen = QPushButton("Generate Stow Strategy")
        btn_gen.clicked.connect(self.run_stow_gen)
        form.addRow(btn_gen)
        
        box.addLayout(form)
        self.layout.addWidget(box)
        
        self.update_stow_ui_visibility()

    def refresh_contour_list(self):
        self.cmb_stg_contour.clear()
        try:
            base = self.pm.base_path
            c_dir = os.path.join(base, "Safe Elevation Contours")
            if os.path.exists(c_dir):
                files = [f for f in os.listdir(c_dir) if f.endswith(".json")]
                self.cmb_stg_contour.addItems(files)
        except: pass

    def update_stow_ui_visibility(self):
        mode = self.cmb_stg_mode.currentText()
        is_contour = (mode == "Clash Contour")
        
        # Counter-Rotation specific
        self.lbl_min_gap.setVisible(not is_contour)
        self.sb_stg_min_gap.setVisible(not is_contour)
        self.lbl_safe_el.setVisible(not is_contour)
        self.sb_stg_safe_el.setVisible(not is_contour)
        self.lbl_offset.setVisible(not is_contour)
        self.sb_stg_offset.setVisible(not is_contour)
        
        # Contour specific
        self.lbl_contour.setVisible(is_contour)
        self.cmb_stg_contour.setVisible(is_contour)
        self.lbl_buffer.setVisible(is_contour)
        self.sb_stg_buffer.setVisible(is_contour)

    def run_stow_gen(self):
        from lcp.analysis.stow_strategy import StowStrategyGenerator
        import shutil
        import json
        from datetime import datetime
        
        sim_name = self.cmb_stg_source.currentText()
        if not sim_name:
            QMessageBox.warning(self, "Input Error", "Please select a source simulation.")
            return

        mode = self.cmb_stg_mode.currentText()
        
        # Common
        speed = self.sb_stg_speed.value()
        
        # Old Params
        min_gap = self.sb_stg_min_gap.value()
        safe_el = self.sb_stg_safe_el.value()
        offset = self.sb_stg_offset.value()
        
        # New Params
        contour_file = self.cmb_stg_contour.currentText()
        el_buffer = self.sb_stg_buffer.value()
        
        contour_path = None
        if mode == "Clash Contour":
            if not contour_file:
                QMessageBox.warning(self, "Input Error", "Please select a Clash Contour Map.")
                return
            contour_path = os.path.join(self.pm.base_path, "Safe Elevation Contours", contour_file)
            if not os.path.exists(contour_path):
                QMessageBox.warning(self, "File Error", "Selected Contour Map not found.")
                return

        try:
             # Construct Paths
             base = self.pm.base_path
             sim_dir = os.path.join(base, sim_name)
             ts_path = os.path.join(sim_dir, "timeseries.csv")
             cfg_path = os.path.join(sim_dir, "config.json")
             res_path = os.path.join(sim_dir, "results.csv")
             
             if not os.path.exists(ts_path):
                 QMessageBox.critical(self, "Error", f"timeseries.csv not found in {sim_name}")
                 return
             
             # Create New Simulation Name (Suffix based on Mode?)
             suffix = "Stow" if mode == "Counter-Rotation" else "ClashStow"
             base_name = f"{sim_name}_{suffix}"
             counter = 1
             while True:
                 new_sim_name = f"{base_name}_v{counter}"
                 new_sim_dir = os.path.join(base, new_sim_name)
                 if not os.path.exists(new_sim_dir):
                     break
                 counter += 1
                 
             os.makedirs(new_sim_dir)
             
             # Copy Config and Results (Summary)
             if os.path.exists(cfg_path):
                 # Load original config
                 with open(cfg_path, 'r') as f:
                     cfg_data = json.load(f)
                 
                 # Inject Stow Parameters
                 stow_cfg = {
                     'strategy_mode': mode,
                     'max_motor_speed_deg_per_min': speed,
                     'source_simulation': sim_name,
                     'generated_at': str(datetime.now())
                 }
                 
                 if mode == "Counter-Rotation":
                     stow_cfg.update({
                         'min_safe_interval_min': min_gap,
                         'safe_stow_el': safe_el,
                         'westward_offset_deg': offset,
                     })
                 else:
                     stow_cfg.update({
                         'contour_map': contour_file,
                         'elevation_buffer_deg': el_buffer
                     })
                     
                 cfg_data['stow_strategy'] = stow_cfg
                 
                 # Save to new location
                 with open(os.path.join(new_sim_dir, "config.json"), 'w') as f:
                     json.dump(cfg_data, f, indent=4)
                     
             if os.path.exists(res_path):
                 shutil.copy2(res_path, os.path.join(new_sim_dir, "results.csv"))
                 
             # Instantiate Generator
             gen = StowStrategyGenerator(
                 strategy_mode=mode,
                 min_safe_interval_min=min_gap,
                 safe_stow_el=safe_el,
                 westward_offset_deg=offset,
                 max_motor_speed_deg_per_min=speed,
                 contour_map_path=contour_path,
                 el_buffer=el_buffer
             )
             
             # Process Timeseries -> New Timeseries
             new_ts_path = os.path.join(new_sim_dir, "timeseries.csv")
             
             # Show Wait Cursor
             QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
             try:
                 gen.process_csv(ts_path, config_path=cfg_path, output_path=new_ts_path)
             finally:
                 QApplication.restoreOverrideCursor()
             
             # Auto-Load
             self.do_load_sim(new_sim_name)
             
             # Switch Mode in Results Widget
             if self.window() and hasattr(self.window(), 'results'):
                  res = self.window().results
                  if res and hasattr(res, 'cmb_mode'):
                       res.cmb_mode.setCurrentText("Stow Strategy")
                       
             QMessageBox.information(self, "Success", f"Stow Strategy Generated!\n\nNew Simulation Created & Loaded: {new_sim_name}")
             
        except Exception as e:
             import traceback
             traceback.print_exc()
             QMessageBox.critical(self, "Generation Failed", str(e))

    def refresh_load_list(self):
        # Only refresh generator source combo
        if hasattr(self, 'cmb_stg_source'):
            self.cmb_stg_source.clear()
            try:
                 sims = self.pm.list_simulations()
                 self.cmb_stg_source.addItems(sims)
            except: pass
            
        # Refresh Contour Map combo
        if hasattr(self, 'cmb_cmap_source'):
            self.cmb_cmap_source.clear()
            try:
                 sims = self.pm.list_simulations()
                 self.cmb_cmap_source.addItems(sims)
            except: pass

    def init_contour_map(self):
        self.box_contour = CollapsibleBox("Clash Contour Mapping")
        layout = QVBoxLayout()
        
        # Select Simulation
        layout.addWidget(QLabel("Source Simulation:"))
        self.cmb_cmap_source = QComboBox()
        try:
            self.cmb_cmap_source.addItems(self.pm.list_simulations())
        except: pass
        layout.addWidget(self.cmb_cmap_source)
        
        # Run Button
        self.btn_run_cmap = QPushButton("Run Mapping")
        self.btn_run_cmap.clicked.connect(self.run_contour_map)
        layout.addWidget(self.btn_run_cmap)
        
        self.box_contour.addLayout(layout)
        self.layout.addWidget(self.box_contour)

    def run_contour_map(self):
        from lcp.analysis.contour_map import ContourMapper
        
        sim_name = self.cmb_cmap_source.currentText()
        if not sim_name:
            QMessageBox.warning(self, "Input Error", "Please select a source simulation.")
            return

        try:
             # Construct Sim Path
             base = self.pm.base_path
             sim_dir = os.path.join(base, sim_name)
             
             # Instantiate Mapper
             # Use a subfolder in base_path called "Safe Elevation Contours"?
             # User requested "new folder called 'Safe Elevation Contours'" saving export.
             # ContourMapper defaults to CWD/Safe Elevation Contours.
             # Let's align it to reside inside the Storage Path for cleanliness? 
             # Or stick to user request which implies a specific folder name. 
             # I'll create it within the storage path to keep things self-contained.
             
             output_dir = os.path.join(base, "Safe Elevation Contours")
             mapper = ContourMapper(output_dir=output_dir)
             
             QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
             try:
                 out_path = mapper.generate_map(sim_dir, sim_name)
             finally:
                 QApplication.restoreOverrideCursor()
                 
             QMessageBox.information(self, "Success", f"Clash Contour generated:\n{out_path}")
             
        except Exception as e:
             import traceback
             traceback.print_exc()
             QMessageBox.critical(self, "Clash Contour Generation Failed", str(e))


        
    def export_stow_strategy(self):
        from lcp.analysis.stow_strategy import StowStrategyGenerator
        
        sim_name = self.lbl_current_sim.text()
        if not sim_name or sim_name == "No Simulation Loaded":
             QMessageBox.warning(self, "Export Error", "No simulation loaded to export.")
             return
             
        base = self.pm.base_path
        sim_dir = os.path.join(base, sim_name)
        ts_path = os.path.join(sim_dir, "timeseries.csv")
        
        if not os.path.exists(ts_path):
             QMessageBox.critical(self, "Export Error", "timeseries.csv not found.")
             return
             
        # Export Name
        export_name = f"{sim_name}_Export.csv"
        # base is already `saved_simulations` (or user configured folder)
        export_path = os.path.join(base, export_name) 
        
        # Sun CSV Dir
        # Assuming fixed path relative to App Root?
        # Need to find project root or use configured path.
        # I'll look for "sun positions CSV" in CWD or up one level.
        sun_dir = "sun positions CSV"
        if not os.path.exists(sun_dir):
             # Try absolute?
             # For now assume running from root.
             pass
             
        try:
             # We use the Generator class just for its static-like export method
             gen = StowStrategyGenerator()
             
             QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
             try:
                 gen.export_strategy_csv(ts_path, export_path, sun_dir)
             finally:
                 QApplication.restoreOverrideCursor()
                 
             QMessageBox.information(self, "Export Success", f"Strategy exported to:\n{export_name}")
             
        except Exception as e:
             import traceback
             traceback.print_exc()
             QMessageBox.critical(self, "Export Failed", str(e))

    def init_visual_settings(self):
         box = CollapsibleBox("3D View Settings", expanded=False)
         form = QFormLayout()
         
         self.cb_show_pivots = QCheckBox("Pivots")
         self.cb_show_pivots.setChecked(True)
         form.addRow(self.cb_show_pivots)
         
         self.cb_show_rays = QCheckBox("Sunrays")
         form.addRow(self.cb_show_rays)
         
         self.cb_show_full = QCheckBox("Full Plant")
         self.cb_show_full.setChecked(False)
         form.addRow(self.cb_show_full)
         
         pass # Toggle removed by user request

         self.cb_show_tolerance = QCheckBox("Panels with Tolerance")
         self.cb_show_tolerance.setChecked(False)
         form.addRow(self.cb_show_tolerance)
         
         # New Feature: Stow All Panels
         self.chk_stow_all = QCheckBox("Stow All Panels")
         self.chk_stow_all.setToolTip("Force all panels (Active & Inactive) to follow the Active/Stow profile.")
         self.chk_stow_all.setChecked(False)
         self.chk_stow_all.toggled.connect(self.stow_all_changed.emit)
         form.addRow(self.chk_stow_all)
         
         box.addLayout(form)
         self.layout.addWidget(box)
        
    def on_sim_change(self):
        s = self.state.sim_settings
        qdate = self.de_start.date()
        s.start_date = date(qdate.year(), qdate.month(), qdate.day())
        s.full_year = self.cb_full.isChecked()
        s.duration_days = self.sb_days.value()
        s.timestep_min = self.sb_step.value()
        s.clashes_only_mode = self.cb_clashes.isChecked()
        s.target_solstice = "Summer (Dec)" # Hardcoded
        s.clash_window = self.sb_window.value()
        s.clash_window = self.sb_window.value()
        s.clash_min_elevation = self.sb_min_el.value()
        
        # Update Sun Source in Config
        self.state.config.sun_source = self.cmb_sun_source.currentText()
        
        is_clash_mode = s.clashes_only_mode
        self.sb_window.setEnabled(is_clash_mode)
        self.sb_min_el.setEnabled(is_clash_mode)
        
        self.de_start.setEnabled(not is_clash_mode and not s.full_year)
        self.sb_days.setEnabled(not s.full_year and not is_clash_mode)
        self.cb_full.setEnabled(not is_clash_mode) 

        if is_clash_mode and s.full_year:
             self.cb_full.setChecked(False) 
             s.full_year = False
        
        if self.cb_full.isChecked():
             self.cb_clashes.setEnabled(False)
        else:
             self.cb_clashes.setEnabled(True)
        self.set_stale()
        
    def set_stale(self):
        """Mark params as changed"""
        self.btn_run.setStyleSheet(self.STYLE_STALE)
        self.btn_run.setText("Input modified - Rerun")
        
    def set_running(self):
        """Mark as running"""
        self.btn_run.setStyleSheet(self.STYLE_RUNNING)
        self.btn_run.setText("Running...")
        self.btn_run.setEnabled(False)
        
    def reset_ready(self):
        """Mark as ready/synced"""
        self.btn_run.setStyleSheet(self.STYLE_READY)
        self.btn_run.setText("▶ RUN SIMULATION")
        self.btn_run.setEnabled(True)
        
    def update_widgets_from_state(self):
        """
        Syncs all input widgets with the current values in self.state.
        Block signals to prevent triggering change events during update.
        """
        # Block Signals
        self.blockSignals(True) 
        
        # Geometry
        g = self.state.geometry
        self.sb_width.blockSignals(True); self.sb_width.setValue(g.width); self.sb_width.blockSignals(False)
        self.sb_length.blockSignals(True); self.sb_length.setValue(g.length); self.sb_length.blockSignals(False)
        self.sb_pivot_depth_glass.blockSignals(True); self.sb_pivot_depth_glass.setValue(self.pivot_depth_glass); self.sb_pivot_depth_glass.blockSignals(False)
        self.sb_connected_thickness.blockSignals(True); self.sb_connected_thickness.setValue(g.thickness); self.sb_connected_thickness.blockSignals(False)
        
        # Config
        c = self.state.config
        self.sb_grid_pitch_x.blockSignals(True); self.sb_grid_pitch_x.setValue(c.grid_pitch_x); self.sb_grid_pitch_x.blockSignals(False)
        self.sb_grid_pitch_y.blockSignals(True); self.sb_grid_pitch_y.setValue(c.grid_pitch_y); self.sb_grid_pitch_y.blockSignals(False)
        self.sb_field_spacing_x.blockSignals(True); self.sb_field_spacing_x.setValue(c.field_spacing_x); self.sb_field_spacing_x.blockSignals(False)
        self.sb_field_spacing_y.blockSignals(True); self.sb_field_spacing_y.setValue(c.field_spacing_y); self.sb_field_spacing_y.blockSignals(False)
        self.sb_tolerance.blockSignals(True); self.sb_tolerance.setValue(c.tolerance); self.sb_tolerance.blockSignals(False)
        self.sb_min_elevation.blockSignals(True); self.sb_min_elevation.setValue(c.min_elevation); self.sb_min_elevation.blockSignals(False)
        
        # Sizing / Config
        self.sb_rows.blockSignals(True); self.sb_rows.setValue(self.state.rows); self.sb_rows.blockSignals(False)
        self.sb_cols.blockSignals(True); self.sb_cols.setValue(self.state.cols); self.sb_cols.blockSignals(False)
        self.sb_rotation.blockSignals(True); self.sb_rotation.setValue(c.plant_rotation); self.sb_rotation.blockSignals(False)
        self.lbl_total.setText(str(c.total_panels))
        
        # Sim Settings
        s = self.state.sim_settings
        self.de_start.blockSignals(True); self.de_start.setDate(QDate(s.start_date.year, s.start_date.month, s.start_date.day)); self.de_start.blockSignals(False)
        self.cb_full.blockSignals(True); self.cb_full.setChecked(s.full_year); self.cb_full.blockSignals(False)
        self.sb_days.blockSignals(True); self.sb_days.setValue(s.duration_days); self.sb_days.blockSignals(False)
        self.sb_step.blockSignals(True); self.sb_step.setValue(s.timestep_min); self.sb_step.blockSignals(False)
        self.sb_step.blockSignals(True); self.sb_step.setValue(s.timestep_min); self.sb_step.blockSignals(False)
        self.cmb_sun_source.blockSignals(True); self.cmb_sun_source.setCurrentText(self.state.config.sun_source); self.cmb_sun_source.blockSignals(False)
        
        self.cb_clashes.blockSignals(True); self.cb_clashes.setChecked(s.clashes_only_mode); self.cb_clashes.blockSignals(False)
        self.sb_window.blockSignals(True); self.sb_window.setValue(s.clash_window); self.sb_window.blockSignals(False)
        self.sb_min_el.blockSignals(True); self.sb_min_el.setValue(s.clash_min_elevation); self.sb_min_el.blockSignals(False)
        
        is_clash = s.clashes_only_mode
        self.sb_window.setEnabled(is_clash)
        self.sb_min_el.setEnabled(is_clash)
        self.sb_days.setEnabled(not s.full_year and not is_clash)
        
        self.blockSignals(False)
