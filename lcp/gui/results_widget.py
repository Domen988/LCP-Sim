from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QPushButton, QSlider, QLabel)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
import pyqtgraph as pg
import pandas as pd
import numpy as np

# Physics Imports
from lcp.physics.engine import InfiniteKernel, PanelState
from lcp.gui.landscape_widget import AnnualLandscapeWidget

class ResultsWidget(QWidget):
    # args: sun_az, sun_el, safety_detected, states_list
    replay_frame = pyqtSignal(float, float, bool, list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = None 
        self.kernel = None 
        self.enable_safety = False # Viz Toggle
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # State
        self.all_data = [] 
        self.current_frames = [] 
        self.replay_idx = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.timer.setInterval(50) 
        
        # Tab 1: Summary Table
        self.table_tab = QWidget()
        self.setup_table_tab()
        self.tabs.addTab(self.table_tab, "Daily Summary")
        
        # Tab 2: Hourly Plot
        self.plot_tab = QWidget()
        self.setup_plot_tab()
        self.tabs.addTab(self.plot_tab, "Power Curve")
        
        # Tab 3: Annual Landscape
        self.landscape = AnnualLandscapeWidget()
        self.tabs.addTab(self.landscape, "Annual Landscape")
        
    def setup_table_tab(self):
        l = QVBoxLayout()
        self.table_tab.setLayout(l)
        
        self.table = QTableWidget()
        self.table.setColumnCount(8) 
        self.table.setHorizontalHeaderLabels([
            "Date", "Theo (MWh)", "Act (MWh)", 
            "Stow Loss %", "Shadow Loss %", 
            "Σ Theo (MWh)", "Σ Act (MWh)", "Efficiency %"
        ])
        
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self.on_day_selected)
        l.addWidget(self.table)
        
    def setup_plot_tab(self):
        l = QVBoxLayout()
        self.plot_tab.setLayout(l)
        
        # Date Axis for X
        axis = pg.AxisItem(orientation='bottom')
        self.chart = pg.PlotWidget(axisItems={'bottom': axis})
        self.chart.setBackground('k')
        self.chart.addLegend()
        self.chart.showGrid(x=True, y=True)
        self.chart.setLabel('left', 'Power (MW)')
        self.chart.setLabel('bottom', 'Time')
        
        self.cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=2))
        self.chart.addItem(self.cursor_line)
        
        l.addWidget(self.chart)
        
        # Replay Controls
        controls = QHBoxLayout()
        
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self.toggle_play)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.on_slider_change)
        
        self.lbl_time = QLabel("--:--")
        
        controls.addWidget(self.btn_play)
        controls.addWidget(self.slider)
        controls.addWidget(self.lbl_time)
        
        l.addLayout(controls)
        
    def update_results(self, data):
        if not data: return
        self.all_data = data
        
        # Update 3D Landscape
        self.landscape.update_data(data)
        
        # 1. Update Table
        self.table.setRowCount(len(data))
        
        cum_theo = 0.0
        cum_act = 0.0
        
        for r, day in enumerate(data):
            summ = day['summary']
            t_mwh = summ['theo_kwh'] / 1000
            a_mwh = summ['act_kwh'] / 1000
            
            cum_theo += t_mwh
            cum_act += a_mwh
            
            stow_pct = (summ['stow_loss_kwh'] / summ['theo_kwh'] * 100) if t_mwh > 0 else 0
            shad_pct = (summ['shad_loss_kwh'] / summ['theo_kwh'] * 100) if t_mwh > 0 else 0
            eff_pct = (cum_act / cum_theo * 100) if cum_theo > 0 else 0
            
            self.table.setItem(r, 0, QTableWidgetItem(str(summ['date'])))
            self.table.setItem(r, 1, QTableWidgetItem(f"{t_mwh:.2f}"))
            self.table.setItem(r, 2, QTableWidgetItem(f"{a_mwh:.2f}"))
            self.table.setItem(r, 3, QTableWidgetItem(f"{stow_pct:.1f}%"))
            self.table.setItem(r, 4, QTableWidgetItem(f"{shad_pct:.1f}%"))
            self.table.setItem(r, 5, QTableWidgetItem(f"{cum_theo:.2f}"))
            self.table.setItem(r, 6, QTableWidgetItem(f"{cum_act:.2f}"))
            self.table.setItem(r, 7, QTableWidgetItem(f"{eff_pct:.1f}%"))
            
        # Select first day by default
        self.table.clearSelection()
        if len(data) > 0:
            self.table.selectRow(0)
        
    def on_day_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows: return
        idx = rows[0].row()
        
        if idx < len(self.all_data):
            self.load_day_frames(self.all_data[idx])
            
    def load_day_frames(self, day_data):
        self.current_frames = day_data['frames']
        if not self.current_frames: return
        
        # Setup Slider
        self.slider.setRange(0, len(self.current_frames) - 1)
        self.slider.setValue(0)
        
        # Setup X Axis Tick Strings
        ticks = []
        for i, f in enumerate(self.current_frames):
             dt = f['time']
             if dt.minute == 0: # Hourly ticks
                  ticks.append((i, dt.strftime("%H:%M")))
        
        ax = self.chart.getAxis('bottom')
        ax.setTicks([ticks])
        
        # Plot
        self.chart.clear()
        self.chart.addItem(self.cursor_line)
        
        x = range(len(self.current_frames))
        y_act = [f['act_w']/1e6 for f in self.current_frames]
        
        if 'theo_w' in self.current_frames[0]:
            y_theo = [f['theo_w']/1e6 for f in self.current_frames]
            self.chart.plot(x, y_theo, pen=pg.mkPen('c', style=Qt.PenStyle.DashLine), name="Theoretical")
            
        self.chart.plot(x, y_act, pen=pg.mkPen('g', width=2), name="Actual Power")
        
        clash_x = []
        clash_y = []
        for i, f in enumerate(self.current_frames):
             if f['safety']:
                  clash_x.append(i)
                  clash_y.append(f['act_w']/1e6)
                  
        if clash_x:
             self.chart.plot(clash_x, clash_y, pen=None, symbol='x', symbolBrush='r', name="Clash")

    def toggle_play(self, checked):
        if checked:
            self.btn_play.setText("⏸ Stop")
            self.timer.start()
        else:
            self.btn_play.setText("▶ Play")
            self.timer.stop()
            
    def next_frame(self):
        val = self.slider.value()
        if val < self.slider.maximum():
            self.slider.setValue(val + 1)
        else:
            self.toggle_play(False)
            self.btn_play.setChecked(False)
            
    def set_enable_safety(self, val):
        self.enable_safety = val
        self.on_slider_change(self.slider.value()) # Refresh

    def on_slider_change(self, val):
        if not self.current_frames or val >= len(self.current_frames): return
        
        frame = self.current_frames[val]
        
        self.cursor_line.setValue(val)
        
        dt = frame['time']
        az = frame['sun_az']
        el = frame['sun_el']
        self.lbl_time.setText(f"{dt.strftime('%H:%M')} | Az: {az:.1f}° El: {el:.1f}°")
        
        # Reconstruct States if missing (for Shadows in Viz)
        states = frame.get('states', [])
        
        # We ALWAYS reconstruct if we have state and kernel, to respect the Safety Toggle
        # (loaded frames might have fixed states, but we want to simulate the toggle?)
        # User implies we should toggle it. 
        # But 'frames' contain pre-calculated safety?
        # If we load a simulation, 'safety' might be baked in.
        # But here we are doing "Feature Parity" viz.
        # If 'states' are missing (Basic CSV), we reconstruct.
        # If 'states' are present (Full Pickle), we might not want to override?
        # But user changed behavior in UI.
        # Let's override if we can reconstruct.
        
        if self.state:
            try:
                if not self.kernel:
                     self.kernel = InfiniteKernel(self.state.geometry, self.state.config)
                     
                new_states, _ = self.kernel.solve_timestep(az, el, enable_safety=self.enable_safety)
                states = new_states
            except Exception as e:
                print(f"Viz Reconstruct Error: {e}")
                # Fallback to frame states if avail
                if not states: states = []
        
        self.replay_frame.emit(az, el, frame['safety'], states)
