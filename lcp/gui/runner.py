
import time
from datetime import datetime, timedelta
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal

from lcp.simulation import SimulationRunner
from lcp.gui.state import AppState

class SimulationWorker(QThread):
    """
    Background worker to run the physics simulation Loop.
    Emits signals for progress and completion.
    """
    
    progress = pyqtSignal(int)      # Percent 0-100
    status = pyqtSignal(str)        # Status Text
    finished_data = pyqtSignal(list) # List of dicts (Results)
    error = pyqtSignal(str)
    
    def __init__(self, state: AppState, start_date: datetime, days: int, parent=None):
        super().__init__(parent)
        self.state = state
        self.start_date = start_date
        self.days = days
        self.days = days
        self._is_running = True
        
    def stop(self):
        self._is_running = False
        
    def run(self):
        try:
            self.status.emit("Initializing Kernel...")
            
            # 1. Setup Runner (headless)
            # Re-use existing `SimulationRunner` logic but adapt loop here?
            # Or assume we replicate dashboard loop logic for fine-grained control?
            # Replicating gives us Progress Bar control.
            
            # Define resource_path helper inside run or globally
            import sys
            import os
            def resource_path(relative_path):
                """ Get absolute path to resource, works for dev and for PyInstaller """
                try:
                    # PyInstaller creates a temp folder and stores path in _MEIPASS
                    base_path = sys._MEIPASS
                except Exception:
                    base_path = os.path.abspath(".")
                return os.path.join(base_path, relative_path)

            # Assuming CSV is in CWD as per dashboard.py, or at root of bundle
            # TODO: Make this configurable in AppState
            csv_path = resource_path("Koster direct normal irradiance_Wh per square meter.csv")
            runner = SimulationRunner(csv_path)
            # Apply Config
            geo = self.state.geometry
            cfg = self.state.config
            
            # Update Kernel with current Geo/Cfg
            import numpy as np
            runner.kernel.geo = geo
            runner.kernel.cfg = cfg
            runner.kernel.collider.geo = geo
            runner.kernel.collider.cfg = cfg
            
            # Recalculate Pivots for new Pitch
            runner.kernel.pivots = {}
            for r in range(3):
                for c in range(3):
                    y = (r - 1) * cfg.grid_pitch_y
                    x = (c - 1) * cfg.grid_pitch_x
                    runner.kernel.pivots[(r,c)] = np.array([float(x), float(y), 0.0])
            
            # Load Data (Spectra/DHI)
            self.status.emit("Loading Solar Data...")
            runner.load_data()
            
            timestep_min = self.state.sim_settings.timestep_min
            steps_per_day = int(24 * 60 / timestep_min)
            total_steps = self.days * steps_per_day
            global_step = 0
            
            all_days = []
            
            # 2. Main Loop
            for d in range(self.days):
                if not self._is_running: break
                
                current_day_date = self.start_date.date() + timedelta(days=d)
                start_dt_iter = datetime.combine(current_day_date, datetime.min.time())
                
                day_metrics = {
                    "date": current_day_date,
                    "theo_kwh": 0.0, "act_kwh": 0.0,
                    "stow_loss_kwh": 0.0, "shad_loss_kwh": 0.0,
                    "clash_count": 0
                }
                day_frames = []
                
                # Iter Frames
                for i in range(steps_per_day):
                    if not self._is_running: break
                    dt = start_dt_iter + timedelta(minutes=timestep_min * i)
                    global_step += 1
                    
                    # Throttle Signals
                    if i % 50 == 0:
                        pct = int(global_step / total_steps * 100)
                        self.progress.emit(pct)
                        self.status.emit(f"Simulating {current_day_date} {dt.strftime('%H:%M')}")
                    
                    # Logic
                    sun = runner.solar.get_position(dt)
                    
                    # Night Skip
                    if sun.elevation <= 0:
                        continue
                        
                    local_az = sun.azimuth - cfg.plant_rotation # PLANT ROTATION
                    
                    # Solve
                    states, safety = runner.kernel.solve_timestep(local_az, sun.elevation, enable_safety=True)
                    dni = runner.get_dni(dt)
                    
                    # Calc Power
                    panel_area = geo.width * geo.length
                    step_theo_w = (cfg.total_panels * panel_area) * dni
                    
                    step_act_w = 0.0
                    step_stow_w = 0.0
                    step_shad_w = 0.0
                    
                    
                    # 2b. Weighted Calculation (Fix for 3x3 Kernel Representation)
                    # We map the 9 kernel states to the full plant geometry.
                    
                    # Counts based on Plant Size
                    rows = cfg.total_panels // self.state.cols # Approximate row count if total_panels used directly? Better use state.rows
                    # Actually cfg doesn't store rows/cols, only total_panels. 
                    # But we have self.state.rows/cols available in AppState but SimulationRunner is standalone...
                    # Wait, SimulationWorker has self.state.
                    
                    n_rows = self.state.rows 
                    n_cols = self.state.cols
                    
                    # Calculate Panel Counts for each Kernel Position (0..8)
                    # 0: TL, 1: Top, 2: TR
                    # 3: L,  4: C,   5: R
                    # 6: BL, 7: Bot, 8: BR
                    
                    counts = [0] * 9
                    
                    # 1. Parity Counts for Plant Regions
                    # Interior
                    cnt_int_even = 0
                    cnt_int_odd = 0
                    # Rows 1..N-2, Cols 1..N-2
                    # Approximation for speed:
                    # Half of (Rows-2)*(Cols-2) is Even, Half is Odd.
                    n_int = max(0, (n_rows - 2) * (n_cols - 2))
                    cnt_int_even = n_int // 2
                    cnt_int_odd = n_int - cnt_int_even
                    
                    # Edges (Top/Bot/Left/Right) - Simplified 50/50 split
                    n_top = max(0, n_cols - 2);     cnt_top_e = n_top // 2;     cnt_top_o = n_top - cnt_top_e
                    n_bot = max(0, n_cols - 2);     cnt_bot_e = n_bot // 2;     cnt_bot_o = n_bot - cnt_bot_e
                    n_left = max(0, n_rows - 2);    cnt_left_e = n_left // 2;   cnt_left_o = n_left - cnt_left_e
                    n_right = max(0, n_rows - 2);   cnt_right_e = n_right // 2; cnt_right_o = n_right - cnt_right_e
                    
                    # 2. Assign to Kernel Indices
                    # Kernel Map:
                    # 0(TL) 1(T) 2(TR)
                    # 3(L)  4(C) 5(R)
                    # 6(BL) 7(B) 8(BR)
                    
                    # Corners (Always 1)
                    counts[0] = 1; counts[2] = 1; counts[6] = 1; counts[8] = 1
                    
                    # Edges
                    # Top Row (Index 1): Represents All Top Edge
                    # But if Index 1 Stows (Parity Odd), we need a proxy for Even.
                    # Proxy: Index 1 is (0,1) -> Odd. 
                    # Neighbors: (0,0), (0,2), (1,1). (1,1) is Center.
                    # We can use Index 4 (Center) as proxy? No, Center is different region.
                    # In 3x3, we don't have enough edge representatives.
                    # However, usually Edge panels map to Index 1.
                    # Strategy: Assign ALL Top to Index 1. 
                    # UNLESS we are in Checkerboard mode?
                    # The Dashboard logic was:
                    # "Even Edges -> Corners (Stowed/Safe)"
                    # "Odd Edges -> Edges (Tracking)"
                    # "Even Int -> Center (Stowed)"
                    # "Odd Int -> Right Edge (Tracking)"
                    
                    # Let's verify `dashboard.py` logic:
                    # w_stow[(0,1)] = cnt_top_odd  (Index 1 gets Odd Top)
                    # w_stow[(0,0)] += cnt_top_even (Index 0/Corner gets Even Top) -> Corners serve as Stow Proxy
                    
                    # Interior:
                    # w_stow[(1,1)] = cnt_int_even (Index 4 gets Even Int)
                    # w_stow[(1,2)] += cnt_int_odd (Index 5/Right Edge gets Odd Int) -> Index 5 serves as Track Proxy
                    
                    # Only apply this splitting if SAFETY is ON (Checkerboard).
                    # If Normal Tracking, everything maps to its geometric representative.
                    
                    # Determine Mode from States?
                    # If any state is STOW, we assume Safety Mode/Checkerboard logic is active.
                    is_safety_mode = any(s.mode == 'STOW' for s in states)
                    
                    if is_safety_mode:
                         # --- SAFETY / STOW WEIGHTS ---
                         
                         # Top (Index 1 is Odd). Even Top -> Index 0.
                         counts[1] = cnt_top_o
                         counts[0] += cnt_top_e
                         
                         # Bot (Index 7 is (2,1) Odd). Even Bot -> Index 6 (BL) or 8 (BR)? 
                         # (2,0) BL is Even. (2,0) is Index 6.
                         counts[7] = cnt_bot_o
                         counts[6] += cnt_bot_e
                         
                         # Left (Index 3 is (1,0) Odd). Even Left -> Index 0 (TL) (0,0) Even.
                         counts[3] = cnt_left_o
                         counts[0] += cnt_left_e
                         
                         # Right (Index 5 is (1,2) Odd). Even Right -> Index 2 (TR) (0,2) Even.
                         counts[5] = cnt_right_o
                         counts[2] += cnt_right_e
                         
                         # Interior (Index 4 is (1,1) Even). Odd Int -> Index 5 (Right Edge) (1,2) Odd.
                         # Dashboard used (1,2) as proxy for Odd Interior.
                         counts[4] = cnt_int_even
                         counts[5] += cnt_int_odd
                    else:
                         # --- NORMAL TRACKING WEIGHTS ---
                         # Geometric mapping
                         counts[1] += n_top
                         counts[7] += n_bot
                         counts[3] += n_left
                         counts[5] += n_right
                         counts[4] += n_int
                    
                    p_one_panel_potential = panel_area * dni
                    
                    for k_idx, s in enumerate(states):
                         if k_idx >= 9: break 
                         
                         count = counts[k_idx]
                         if count > 0:
                              # 1. Actual
                              step_act_w += (p_one_panel_potential * s.power_factor) * count
                              
                              # 2. Stow Loss
                              step_stow_w += (p_one_panel_potential * s.stow_loss) * count
                              
                              # 3. Shadow Loss
                              # s.shadow_loss is fractional loss.
                              cos_theta = 1.0 - s.theta_loss
                              step_shad_w += (p_one_panel_potential * cos_theta * s.shadow_loss) * count
                              
                    # Accumulate
                    day_metrics["theo_kwh"] += step_theo_w * (timestep_min/60) / 1000
                    day_metrics["act_kwh"] += step_act_w * (timestep_min/60) / 1000
                    day_metrics["stow_loss_kwh"] += step_stow_w * (timestep_min/60) / 1000
                    day_metrics["shad_loss_kwh"] += step_shad_w * (timestep_min/60) / 1000
                    
                    if safety: day_metrics["clash_count"] += 1
                    
                    day_frames.append({
                         "time": dt,
                         "sun_az": sun.azimuth,
                         "sun_el": sun.elevation,
                         "dni": dni,
                         "theo_w": step_theo_w,
                         "act_w": step_act_w,
                         "stow_w": step_stow_w,
                         "shad_w": step_shad_w,
                         "safety": safety,
                         "states": states
                    })
                
                # End Day
                all_days.append({
                     "summary": day_metrics,
                     "frames": day_frames
                })
                
            self.progress.emit(100)
            self.status.emit("Complete")
            self.finished_data.emit(all_days)
            
        except Exception as e:
            self.error.emit(str(e))
            
    def stop(self):
        self._is_running = False
