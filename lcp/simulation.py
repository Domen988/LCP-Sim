import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from scipy.interpolate import CubicSpline

from lcp.core.geometry import PanelGeometry
from lcp.core.config import ScenarioConfig
from lcp.core.solar import SolarCalculator
from lcp.physics.engine import InfiniteKernel

class SimulationRunner:
    def __init__(self, weather_path):
        self.weather_path = weather_path
        self.geo = PanelGeometry()
        self.cfg = ScenarioConfig()
        self.kernel = InfiniteKernel(self.geo, self.cfg)
        self.solar = SolarCalculator()
        self.splines = {} # Cache for CubicSplines
        
    def load_data(self) -> pd.DataFrame:
        """
        Loads the Monthly/Hourly matrix and builds Cubic Splines for interpolation.
        Returns the raw DF for debug.
        """
        df = pd.read_csv(self.weather_path, sep=';')
        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        
        # Build Splines for each Month
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        hours = np.arange(24) # 0 to 23
        
        for i, m_col in enumerate(month_names):
            if m_col in df.columns:
                values = df[m_col].values[:24] # Assume first 24 rows are hours
                # BC Type 'clamped' forces derivative to 0 at ends (Midnight)
                # Ensure values allow for correct spline (float)
                y = values.astype(float)
                self.splines[i] = CubicSpline(hours, y, bc_type='clamped')
        
        # Pre-calc anchor days (15th of each month)
        # Non-leap year 2025
        self.anchor_days = []
        for m in range(1, 13):
            self.anchor_days.append(datetime(2025, m, 15).timetuple().tm_yday)
            
        return df

    def get_dni(self, dt: datetime, data: pd.DataFrame = None) -> float:
        """
        Interpolates DNI using Cubic Spline, smoothing day-to-day between months.
        Assumes monthly data corresponds to the 15th of each month.
        """
        # Ensure splines are loaded
        if not self.splines:
             # Just fallback or error, but assuming loaded
             return 0.0

        doy = dt.timetuple().tm_yday
        h = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        
        # Find anchors
        # Safe defaults
        prev_idx = -1
        next_idx = -1
        alpha = 0.0
        
        # Edge Case: Before Jan 15 -> Interpolate Dec to Jan
        if doy < self.anchor_days[0]:
            prev_idx = 11 # Dec
            next_idx = 0  # Jan
            # Dec 15 is roughly day -16 relative to Jan 1
            # Distance from Dec 15 to Jan 15 is approx 31 days
            # alpha = (doy + (365 - self.anchor_days[11])) / (self.anchor_days[0] + (365 - self.anchor_days[11]))
            # Easier:
            # Span = (365 - 349) + 15 = 16 + 15 = 31 days
            # Progress = (365 - 349) + doy = 16 + doy
            span = (365 - self.anchor_days[11]) + self.anchor_days[0]
            progress = (365 - self.anchor_days[11]) + doy
            alpha = progress / span
            
        # Edge Case: After Dec 15 -> Interpolate Dec to Jan
        elif doy >= self.anchor_days[-1]:
            prev_idx = 11 # Dec
            next_idx = 0  # Jan
            # Span same as above (31 days)
            # Progress = doy - 349
            span = (365 - self.anchor_days[11]) + self.anchor_days[0]
            progress = doy - self.anchor_days[11]
            alpha = progress / span
            
        else:
            # Normal Case: Between two months
            for i in range(len(self.anchor_days) - 1):
                if self.anchor_days[i] <= doy < self.anchor_days[i+1]:
                    prev_idx = i
                    next_idx = i + 1
                    span = self.anchor_days[i+1] - self.anchor_days[i]
                    progress = doy - self.anchor_days[i]
                    alpha = progress / span
                    break
        
        # Evaluate Splines
        val_prev = max(0.0, float(self.splines[prev_idx](h)))
        val_next = max(0.0, float(self.splines[next_idx](h)))
        
        # Weighted Average
        # if alpha 0 -> prev, alpha 1 -> next
        val = (1.0 - alpha) * val_prev + alpha * val_next
        
        return val

    def run_year(self) -> pd.DataFrame:
        """
        Runs the simulation for a full year (or subset).
        """
        # ... (Existing logic, but relying on get_dni which now uses internal splines)
        # Note: run_year isn't used by the dashboard directly (dashboard has its own loop).
        pass  # Dashboard implements its own loop loop currently.

        """
        Runs the simulation for a full year (or subset).
        """
        data = self.load_data()
        
        # Generate timestamps
        start = datetime(2025, 1, 1)
        end = datetime(2025, 12, 31, 23, 50)
        # 6 min steps
        steps = pd.date_range(start, end, freq='6min')
        
        results = []
        
        print(f"Starting Simulation for {len(steps)} steps...")
        
        # We can optimize by not running night steps?
        # But we need the full series for the UI slider.
        
        # Bias from Spec: Plant Rotated +5 deg Clockwise.
        # Sun Az relative to Plant = Sun Az (Global) - 5 deg.
        # If Plant N is rotated +5 deg (to N' = 5 deg).
        # A sun at 5 deg (True N) should match Plant N (0 deg local).
        # So Local Az = True Az - 5. Correct.
        # LOCAL AZIMUTH: Sun Az (Global) - Plant Rotation
        # plant_rotation=5 means plant North is 5 deg East of True North.
        # To align, we subtract 5 deg from Sun Az.
        PLANT_ROTATION = -self.cfg.plant_rotation 
        
        for dt in steps:
            # 1. Physics
            sun = self.solar.get_position(dt)
            local_az = sun.azimuth - PLANT_ROTATION
            
            # NIGHT OPTIMIZATION
            if sun.elevation <= 0:
                 results.append({
                    "Timestamp": dt,
                    "Sun_Az": sun.azimuth,
                    "Sun_El": sun.elevation,
                    "Local_Az": local_az,
                    "DNI": 0.0,
                    "Safety_Mode": True, # Safe
                    "Theo_Power": 0.0,
                    "Actual_Power": 0.0,
                    "Stow_Loss": 0.0,
                    "Shadow_Loss": 0.0
                })
                 continue

            states, safety_triggered = self.kernel.solve_timestep(local_az, sun.elevation)
            
            # 2. Power Check
            dni = self.get_dni(dt, data)
            
            # Aggregate stats from Kernel (3x3 = 9 panels)
            # We want "Total Power" scaling strategy.
            # Spec: "Extrapolation: Results from 3x3 Kernel used to calculate total."
            # Let's just output the Kernel Sum for now, UI can scale.
            
            # Theoretical Power (All tracking perfectly)
            # Area * DNI * Cos(Theta_Ideal)
            # Ideal Theta is just Sun Normal vs Panel (assuming perfect track).
            # Perfect track -> Normal // Sun -> Cos=1.
            # So Theoretical = Area * DNI.
            
            # Scale to Whole Plant
            # Extrapolate from 9-panel kernel
            n_kernel = 9.0
            total_panels = self.cfg.total_panels
            panel_area = self.geo.width * self.geo.length
            
            # Aggregate Kernel Metrics
            k_act_sum = 0.0
            k_stow_sum = 0.0
            k_shad_sum = 0.0
            
            for s in states:
                k_act_sum += s.power_factor
                k_stow_sum += s.stow_loss
                # Weight shadow loss by cosine factor for energy realism
                cos_theta = 1.0 - s.theta_loss
                k_shad_sum += (s.shadow_loss * cos_theta)

            p_theo_plant = (panel_area * total_panels) * dni
            
            # Theoretical Kernel Power (if all 9 panels were perfect 1.0)
            # But we want to scale checking the "average per panel" in kernel.
            
            p_act_plant = (k_act_sum / n_kernel) * p_theo_plant
            p_stow_plant = (k_stow_sum / n_kernel) * p_theo_plant
            p_shad_plant = (k_shad_sum / n_kernel) * p_theo_plant
            
            # Store Row
            res = {
                "Timestamp": dt,
                "Sun_Az": sun.azimuth,
                "Sun_El": sun.elevation,
                "Local_Az": local_az,
                "DNI": dni,
                "Safety_Mode": safety_triggered,
                "Theo_Power": p_theo_plant,
                "Actual_Power": p_act_plant,
                "Stow_Loss": p_stow_plant,
                "Shadow_Loss": p_shad_plant
            }
            results.append(res)
            
        return pd.DataFrame(results)
