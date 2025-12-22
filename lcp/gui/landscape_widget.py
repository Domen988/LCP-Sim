
import numpy as np
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame
from datetime import datetime
import pandas as pd

class AnnualLandscapeWidget(QWidget):
    """
    Container for 3D Surface Plot + 2D Overlay (Legend).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        
        # 3D View
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=220, elevation=35, azimuth=45)
        self.view.setBackgroundColor((30, 30, 30, 255))
        self.layout.addWidget(self.view)
        
        # 2D Legend Overlay
        self.overlay = QWidget(self.view)
        self.overlay.setStyleSheet("background-color: rgba(0, 0, 0, 150); border-radius: 5px; padding: 5px;")
        l_ov = QVBoxLayout(self.overlay)
        
        self.lbl_title = QLabel("Annual Power Landscape")
        self.lbl_title.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")
        l_ov.addWidget(self.lbl_title)
        
        # Legend Items
        def add_leg(color, text):
            row = QHBoxLayout()
            box = QLabel()
            box.setFixedSize(15, 15)
            box.setStyleSheet(f"background-color: {color}; border: 1px solid white;")
            lbl = QLabel(text)
            lbl.setStyleSheet("color: white; font-size: 12px;")
            row.addWidget(box)
            row.addWidget(lbl)
            row.addStretch()
            l_ov.addLayout(row)
            
        add_leg("qlineargradient(stop:0 blue, stop:1 yellow)", "Actual Power (Surface)")
        add_leg("white", "Theoretical Max (Lines)")
        
        self.overlay.adjustSize()
        self.overlay.move(20, 20)
        
        # --- SCENE ITEMS ---
        # Grid
        g = gl.GLGridItem()
        g.setSize(x=100, y=100, z=0)
        g.setSpacing(x=10, y=10, z=0)
        self.view.addItem(g)
        
        # Axis Lines
        # self.view.addItem(gl.GLAxisItem(size=gl.QVector3D(50,50,50)))
        
        # Placeholders
        self.surface = None
        self.theo_lines = None
        self.label_items = []
        
        self.z_data = None
        self.x_times = None 
        
    def update_data(self, results):
        if not results: return
        
        # Clear Old
        if self.surface: 
            self.view.removeItem(self.surface)
            self.surface = None
        if self.theo_lines:
            self.view.removeItem(self.theo_lines)
            self.theo_lines = None
        for it in self.label_items:
            self.view.removeItem(it)
        self.label_items = []
            
        # 1. Process Data
        z_act_list = []
        z_theo_list = []
        y_dates = []
        x_times = None 
        
        # Find Ref Day
        max_valid = 0
        ref_day_idx = -1
        for i, day in enumerate(results):
             frames = day.get('frames', [])
             cnt = len([f for f in frames if 5 <= f['time'].hour <= 20])
             if cnt > max_valid:
                  max_valid = cnt
                  ref_day_idx = i
        
        if ref_day_idx == -1: return 
        
        ref_frames = [f for f in results[ref_day_idx]['frames'] if 5 <= f['time'].hour <= 20]
        x_times = [f['time'].strftime("%H:%M") for f in ref_frames]
        
        # Build Matrices
        for day in results:
             d_str = day['summary']['date'].strftime("%Y-%m-%d")
             y_dates.append(d_str)
             
             frames = day.get('frames', [])
             valid_frames = [f for f in frames if 5 <= f['time'].hour <= 20]
             
             if not valid_frames:
                  z_act_list.append([0.0]*len(x_times))
                  z_theo_list.append([0.0]*len(x_times))
                  continue
                  
             curr_times = [f['time'].strftime("%H:%M") for f in valid_frames]
             curr_act = [f['act_w']/1000.0 for f in valid_frames]
             curr_theo = [f['theo_w']/1000.0 for f in valid_frames]
             
             if len(curr_times) == len(x_times):
                  z_act_list.append(curr_act)
                  z_theo_list.append(curr_theo)
             else:
                  # Align
                  map_a = dict(zip(curr_times, curr_act))
                  map_t = dict(zip(curr_times, curr_theo))
                  z_act_list.append([map_a.get(t, 0.0) for t in x_times])
                  z_theo_list.append([map_t.get(t, 0.0) for t in x_times])
                  
        z_act = np.array(z_act_list)
        z_theo = np.array(z_theo_list)
        
        # 2. Scaling
        # Target Dimensions in 3D Space: 100 x 100 x 50
        nx = z_act.shape[1] # Time steps
        ny = z_act.shape[0] # Days
        
        if nx == 0 or ny == 0: return
        
        max_p = max(np.max(z_act), np.max(z_theo)) if (np.max(z_act)>0 or np.max(z_theo)>0) else 1.0
        
        sx = 100.0 / nx
        sy = 100.0 / ny
        sz = 50.0 / max_p
        
        # 3. Surface (Actual)
        x_vals = np.linspace(-50, 50, nx)
        y_vals = np.linspace(-50, 50, ny)
        z_grid = z_act * sz
        
        # Colors: Map Z to Blue-Yellow
        colors = np.zeros((nx, ny, 4))
        norm_z = np.clip(z_act / max_p, 0, 1)
        
        for i in range(nx): # Time
            for j in range(ny): # Day
                v = norm_z[j, i] # Raw data is (Day, Time)
                # Simple heatmap: Blue(Low) -> Green -> Yellow(High)
                r = v
                g = v * 0.8 + 0.2
                b = 1.0 - v
                colors[i, j] = [r, g, b, 1.0]
        
        # Transpose Z for GLSurfacePlot (Expects (Nx, Ny))
        self.surface = gl.GLSurfacePlotItem(
             x=x_vals, y=y_vals, z=z_grid.T,
             colors=colors, shader='shaded', smooth=True
        )
        self.view.addItem(self.surface)
        
        # 4. Theoretical Lines
        # We draw one line per day.
        # To batch, we use mode='lines' (segments). 
        # But we want continuous curves per day. 
        # Actually GLLinePlotItem is fast enough with many segments.
        # Let's create `pos` array with NaN separators if possible? OpenGL usually no.
        # We will iterate and create a few combined MultiLines. 
        # Actually, let's just sample every 5th day to avoid clutter or draw all if low count.
        # Drawing 365 lines is fine for desktop GPU.
        
        line_pos = []
        # Center shift (-50, -50) matches x_vals, y_vals range
        # x_vals is array (-50..50).
        # y_vals is array (-50..50).
        # We need grid of (x, y, z)
        
        # Create a single array of points
        # To make disjoint lines, we can use GL_LINES but that requires duplicating points.
        # Or just use many GLLinePlotItems? Too much overhead.
        # Best: Use one GLLinePlotItem per month? Or one big one with 'line_strip' but that connects days.
        # TRICK: Add point at (0,0,-9999) to hide connection? No.
        # Let's use `mode='lines'` -> segments.
        # For each day: (p0, p1), (p1, p2), (p2, p3)...
        # Loops: Days
        
        all_segs = []
        step_day = max(1, ny // 60) # Don't draw every single day if > 60 days
        
        for j in range(0, ny, step_day):
             y = y_vals[j]
             z_row = z_theo[j] * sz
             
             # Create points for this day
             # X varies, Y cons, Z varies
             pts = np.column_stack((x_vals, np.full(nx, y), z_row))
             
             # Segments
             # p[0]..p[N-1]
             # We need p[0],p[1], p[1],p[2], ...
             # Use numpy magic
             p_start = pts[:-1]
             p_end = pts[1:]
             
             # Interleave
             segs = np.empty((p_start.shape[0] * 2, 3))
             segs[0::2] = p_start
             segs[1::2] = p_end
             all_segs.append(segs)
             
        if all_segs:
             total_pts = np.vstack(all_segs)
             self.theo_lines = gl.GLLinePlotItem(
                  pos=total_pts, 
                  color=(1, 1, 1, 0.5), 
                  width=1, 
                  mode='lines',
                  antialias=True
             )
             self.view.addItem(self.theo_lines)
             
        # 5. Labels (Axis)
        # Y-Axis (Dates)
        # Strategy: If dense (<60 days), show every day. Else show ~30 intervals.
        stride = 1 if ny <= 60 else (ny // 30)
        
        for j in range(0, ny, stride):
             y = y_vals[j]
             date_str = pd.to_datetime(y_dates[j]).strftime("%b %d")
             
             # Align text to the left of the graph
             t = gl.GLTextItem(pos=[-55, y, 0], text=date_str, color=(1,1,1,1))
             self.view.addItem(t)
             self.label_items.append(t)
             
        # Z-Axis (Power): Max
        t_max = gl.GLTextItem(pos=[-50, -50, 55], text=f"{max_p:.1f} kW", color=(1,1,0,1))
        self.view.addItem(t_max)
        self.label_items.append(t_max)
        
        # X-Axis (Time): Start/End
        t_start = gl.GLTextItem(pos=[-50, -60, 0], text=x_times[0], color=(1,1,1,1))
        t_end = gl.GLTextItem(pos=[50, -60, 0], text=x_times[-1], color=(1,1,1,1))
        self.view.addItem(t_start)
        self.view.addItem(t_end)
        self.label_items.extend([t_start, t_end])
