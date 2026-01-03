
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QPushButton, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QHBoxLayout, QLabel,
                             QAbstractItemView)
from PyQt6.QtCore import Qt
from typing import List, Dict, Any

class SimulationLoadDialog(QDialog):
    def __init__(self, simulations: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Simulation")
        self.resize(800, 500)
        
        self.selected_sim = None
        self.simulations = simulations
        
        layout = QVBoxLayout(self)
        
        # Header
        layout.addWidget(QLabel("Select a simulation to load:"))
        
        # Table
        self.table = QTableWidget()
        headers = [
            "Name", "Type", "Source Sim", "Date", "Duration", "Sun Source",
            "Tol (m)", "Rot (°)", "Step (m)", 
            "Gap (m)", "Safe El", "Offset", "Speed"
        ]
        tooltips = [
            "Unique name of the simulation",
            "Simulation Type: Standard or Stow Strategy",
            "Parent simulation used to generate this stow strategy",
            "Date and time of creation",
            "Duration of the simulation in days",
            "Source of sun position data (pvlib/calc or csv/file)",
            "Tolerance distance used for collision detection (meters)",
            "Plant rotation angle relative to North (degrees, clockwise)",
            "Simulation timestep interval (minutes)",
            "Minimum Safe Interval: Time to keep panels stowed before/after a clash (minutes)",
            "Safe Elevation angle used for stow position (degrees)",
            "Westward Offset: Azimuth offset for stow strategy generation (degrees)",
            "Maximum motor tracking speed allowed (degrees/minute)"
        ]
        
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        
        # Set Tooltips
        for i, tip in enumerate(tooltips):
            item = self.table.horizontalHeaderItem(i)
            if item:
                item.setToolTip(tip)
        
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        # self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch) # Name stretches
        
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(self.accept_selection)
        
        self.populate_table()
        layout.addWidget(self.table)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        self.btn_load = QPushButton("Load Selected")
        self.btn_load.clicked.connect(self.accept_selection)
        self.btn_load.setEnabled(False)
        self.btn_load.setDefault(True)
        
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(self.btn_load)
        
        layout.addLayout(btn_layout)
        
    def populate_table(self):
        self.table.setRowCount(len(self.simulations))
        for i, sim in enumerate(self.simulations):
            # Name
            self.table.setItem(i, 0, QTableWidgetItem(sim.get('name', '')))
            
            # Type
            self.table.setItem(i, 1, QTableWidgetItem(sim.get('type', '')))
            
            # Source Sim
            self.table.setItem(i, 2, QTableWidgetItem(sim.get('source_sim', '-')))
            
            # Date
            date_str = sim.get('timestamp', '')
            try:
                if '.' in date_str: date_str = date_str.split('.')[0]
                date_str = date_str.replace('T', ' ')
            except: pass
            self.table.setItem(i, 3, QTableWidgetItem(date_str))
            
            # Duration
            self.table.setItem(i, 4, QTableWidgetItem(str(sim.get('duration', ''))))
            
            # Sun Source
            self.table.setItem(i, 5, QTableWidgetItem(sim.get('sun_source', '')))
            
            # Extended Cols
            self.table.setItem(i, 6, QTableWidgetItem(str(sim.get('tolerance', ''))))
            self.table.setItem(i, 7, QTableWidgetItem(str(sim.get('rotation', ''))))
            self.table.setItem(i, 8, QTableWidgetItem(str(sim.get('timestep', ''))))
            
            stow_gap = str(sim.get('stow_gap', ''))
            if stow_gap == '-': stow_gap = ''
            self.table.setItem(i, 9, QTableWidgetItem(stow_gap))
            
            self.table.setItem(i, 10, QTableWidgetItem(str(sim.get('stow_el', ''))))
            self.table.setItem(i, 11, QTableWidgetItem(str(sim.get('stow_offset', ''))))
            self.table.setItem(i, 12, QTableWidgetItem(str(sim.get('stow_speed', ''))))
            
    def on_selection_changed(self):
        self.btn_load.setEnabled(len(self.table.selectedItems()) > 0)
        
    def accept_selection(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            row_idx = rows[0].row()
            self.selected_sim = self.simulations[row_idx]['name']
            self.accept()
