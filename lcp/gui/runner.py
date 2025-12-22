
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
        self._is_running = True
        
    def run(self):
        try:
            self.status.emit("Initializing Kernel...")
            
            # 1. Setup Runner (headless)
            # Re-use existing `SimulationRunner` logic but adapt loop here?
            # Or assume we replicate dashboard loop logic for fine-grained control?
            # Replicating gives us Progress Bar control.
            
            # Assuming CSV is in CWD as per dashboard.py
            # TODO: Make this configurable in AppState
            csv_path = "Koster direct normal irradiance_Wh per square meter.csv"
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
                        
                    local_az = sun.azimuth - 5.0 # PLANT ROTATION
                    
                    # Solve
                    states, safety = runner.kernel.solve_timestep(local_az, sun.elevation, enable_safety=True)
                    dni = runner.get_dni(dt)
                    
                    # Calc Power
                    panel_area = geo.width * geo.length
                    step_theo_w = (cfg.total_panels * panel_area) * dni
                    
                    step_act_w = 0.0
                    step_stow_w = 0.0
                    step_shad_w = 0.0
                    
                    
                    # Simplified Accounting:
                    # The 9-panel kernel represents the average behavior of the plant.
                    count_per_state = cfg.total_panels / len(states)
                    p_one_panel_potential = panel_area * dni # Watts incident if normal to sun
                    
                    for s in states:
                         # 1. Actual Power
                         p_act = p_one_panel_potential * s.power_factor
                         step_act_w += p_act * count_per_state
                         
                         # 2. Stow Loss
                         # If stowed, s.stow_loss is 1.0 (full loss), otherwise 0.0
                         step_stow_w += (p_one_panel_potential * s.stow_loss) * count_per_state
                         
                         # 3. Shadow Loss
                         # Loss = Potential * CosTheta * ShadowFraction
                         # s.point_factor already includes CosTheta * (1-Shadow).
                         # We want the portion lost to shadow.
                         cos_theta = 1.0 - s.theta_loss
                         step_shad_w += (p_one_panel_potential * cos_theta * s.shadow_loss) * count_per_state
                              
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
                         "theo_w": step_theo_w,
                         "act_w": step_act_w,
                         "safety": safety
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
