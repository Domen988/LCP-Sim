import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from scipy.interpolate import CubicSpline

from lcp.core.geometry import PanelGeometry
from lcp.core.config import ScenarioConfig
from lcp.core.solar import SolarCalculator
from lcp.physics.engine import Kernel3x3, PanelState

class SimulationRunner:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.geo = PanelGeometry()
        self.cfg = ScenarioConfig()
        self.kernel = Kernel3x3(self.geo, self.cfg)
        self.solar = SolarCalculator()
        self.splines = {} # Cache for CubicSplines
        
    def load_data(self) -> pd.DataFrame:
        """
        Loads the Monthly/Hourly matrix and builds Cubic Splines for interpolation.
        Returns the raw DF for debug.
        """
        df = pd.read_csv(self.csv_path, sep=';')
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
                
        return df

    def get_dni(self, dt: datetime, data: pd.DataFrame = None) -> float:
        """
        Interpolates DNI using Cubic Spline.
        """
        m_idx = dt.month - 1
        
        if m_idx in self.splines:
            # Fractional Hour
            h = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
            val = self.splines[m_idx](h)
            return max(0.0, float(val))
            
        # Fallback (Legacy)
        if data is not None:
             month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
             col = month_names[m_idx]
             try:
                return float(data.iloc[dt.hour][col])
             except:
                return 0.0
        return 0.0

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
        PLANT_ROTATION = 5.0 
        
        for dt in steps:
            # 1. Physics
            sun = self.solar.get_position(dt)
            local_az = sun.azimuth - PLANT_ROTATION
            
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
            
            total_area = 9.0 * (self.geo.width * self.geo.length) # 9 sqm
            
            # Actual Power
            # Sum of (Area * DNI * power_factor)
            actual_p = 0.0
            stow_loss_p = 0.0
            shad_loss_p = 0.0
            
            for s in states:
                # p_factor = cos * (1-shadow) [0 if stow]
                # We need to break down losses.
                
                # Theoretical for this panel
                p_theo = (self.geo.width * self.geo.length) * dni
                
                if s.mode == "STOW":
                    # All potential is lost to Stow
                    # Except... wait. Stow angle might catch some light?
                    # Spec: "Stowed panels generate 0 Watts."
                    loss = p_theo
                    stow_loss_p += loss
                else:
                    # Tracking
                    # Loss due to Cosine? 
                    # Spec: "Theoretical Power... max potential if tracking perfectly (CosLoss included)."
                    # Wait, usually Theoretical includes Cos Loss relative to generic flat?
                    # "Theoretical Power (kW): The max potential if tracking perfectly."
                    # If tracking perfectly, Cos=1. (Normal points to sun).
                    # So Theoretical = DNI * Area.
                    
                    # Actual = DNI * Area * Cos(alignment_error) * (1-Shadow).
                    # My Rig moves perfectly to Sun. So Cos=1.
                    # Unless limits hit? (My rig has no limits yet).
                    
                    val = p_theo * s.power_factor
                    actual_p += val
                    
                    # Shadow Loss
                    # shad_loss matches fraction of area?
                    # loss = p_theo * s.shadow_loss
                    shad_loss_p += (p_theo * s.shadow_loss)
            
            # Store Row
            res = {
                "Timestamp": dt,
                "Sun_Az": sun.azimuth,
                "Sun_El": sun.elevation,
                "Local_Az": local_az,
                "DNI": dni,
                "Safety_Mode": safety_triggered,
                "Theo_Power": total_area * dni,
                "Actual_Power": actual_p,
                "Stow_Loss": stow_loss_p,
                "Shadow_Loss": shad_loss_p
            }
            results.append(res)
            
        return pd.DataFrame(results)
