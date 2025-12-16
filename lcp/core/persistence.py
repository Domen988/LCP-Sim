
import json
import dataclasses
import pandas as pd
import os
import shutil
from typing import Tuple, Dict, Any, List
from datetime import datetime
from lcp.core.geometry import PanelGeometry
from lcp.core.config import ScenarioConfig

class PersistenceManager:
    """
    Handles saving and loading of simulation configurations and results.
    Stores data in a local directory structure:
    saved_simulations/
      {SimName}/
        config.json
        results.csv
    """
    
    def __init__(self, base_path: str = "saved_simulations"):
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def list_simulations(self) -> List[str]:
        """Returns a list of available simulation names."""
        if not os.path.exists(self.base_path):
            return []
        
        sims = []
        for name in os.listdir(self.base_path):
            path = os.path.join(self.base_path, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
                sims.append(name)
        return sorted(sims)

    def save_simulation(self, name: str, geo: PanelGeometry, cfg: ScenarioConfig, results: List[Dict[str, Any]]) -> str:
        """
        Saves the simulation to a directory.
        """
        # Sanitize name
        clean_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip()
        if not clean_name:
            raise ValueError("Invalid simulation name")
            
        sim_dir = os.path.join(self.base_path, clean_name)
        if not os.path.exists(sim_dir):
            os.makedirs(sim_dir)
            
        # 1. Save Configuration
        data = {
            "geometry": dataclasses.asdict(geo),
            "config": dataclasses.asdict(cfg),
            "version": "1.0",
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(sim_dir, "config.json"), "w") as f:
            json.dump(data, f, indent=4)
            
        # 2. Save Results (Summary only)
        summary_rows = []
        TimeSeriesRow = Any # Type hint helper
        ts_rows = []
        
        for r in results:
            s = r['summary']
            # Make sure date is serialized
            row = s.copy()
            if isinstance(row.get('date'), (datetime, pd.Timestamp)):
                row['date'] = row['date'].isoformat()
            summary_rows.append(row)
            
            # 3. Collect Time Series (Frames without States)
            for f in r.get('frames', []):
                ts_rows.append({
                    "time": f['time'].isoformat(),
                    "sun_az": f['sun_az'],
                    "sun_el": f['sun_el'],
                    "safety": f['safety'],
                    "act_w": f['act_w'],
                    "theo_w": f['theo_w']
                })
            
        df_sum = pd.DataFrame(summary_rows)
        df_sum.to_csv(os.path.join(sim_dir, "results.csv"), index=False)
        
        df_ts = pd.DataFrame(ts_rows)
        df_ts.to_csv(os.path.join(sim_dir, "timeseries.csv"), index=False)
        
        return clean_name

    def load_simulation(self, name: str) -> Tuple[PanelGeometry, ScenarioConfig, List[Dict[str, Any]]]:
        """
        Loads a simulation. Returns (Geo, Cfg, Results).
        Note: The returned results will have 'frames' populated with metrics but 'states' will be empty.
        """
        sim_dir = os.path.join(self.base_path, name)
        if not os.path.exists(sim_dir):
            raise FileNotFoundError(f"Simulation {name} not found")
            
        # 1. Load Config
        with open(os.path.join(sim_dir, "config.json"), "r") as f:
            data = json.load(f)
            
        # Geometry
        geo_data = data.get("geometry", {})
        if "pivot_offset" in geo_data and isinstance(geo_data["pivot_offset"], list):
            geo_data["pivot_offset"] = tuple(geo_data["pivot_offset"])   
        geo = PanelGeometry(**geo_data)
        
        # Config
        cfg_data = data.get("config", {})
        cfg = ScenarioConfig(**cfg_data)
        
        # 2. Load Results Summary
        results = []
        res_path = os.path.join(sim_dir, "results.csv")
        ts_path = os.path.join(sim_dir, "timeseries.csv")
        
        # Load Time Series first if available (for fast lookup)
        ts_df = pd.DataFrame()
        if os.path.exists(ts_path):
            ts_df = pd.read_csv(ts_path)
            ts_df['time'] = pd.to_datetime(ts_df['time'])
            
        if os.path.exists(res_path):
            df = pd.read_csv(res_path)
            for _, row in df.iterrows():
                summary = row.to_dict()
                # Parse date logic
                d_val = summary.get('date')
                dt_obj = None
                if isinstance(d_val, (datetime, pd.Timestamp)):
                    dt_obj = d_val
                elif isinstance(d_val, str):
                    try:
                        dt_obj = datetime.fromisoformat(d_val)
                    except ValueError:
                        try:
                            dt_obj = pd.to_datetime(d_val).to_pydatetime()
                        except:
                            pass
                
                if dt_obj is None:
                    continue # Skip invalid row
                
                summary['date'] = dt_obj
                
                # Reconstruct Frames using TimeSeries data falling on this date
                frames = []
                if not ts_df.empty:
                    # Filter for this day
                    # Normalize comparison to python date objects
                    current_date = dt_obj.date()
                    
                    # Create a boolean mask using dates
                    # Accessing .dt.date results in python date objects
                    mask = ts_df['time'].apply(lambda x: x.date() if isinstance(x, (datetime, pd.Timestamp)) else None) == current_date
                    day_ts = ts_df[mask]
                    
                    for _, ts_row in day_ts.iterrows():
                        frames.append({
                            "time": ts_row['time'].to_pydatetime(),
                            "sun_az": float(ts_row['sun_az']),
                            "sun_el": float(ts_row['sun_el']),
                            "safety": bool(ts_row['safety']),
                            "act_w": float(ts_row['act_w']),
                            "theo_w": float(ts_row['theo_w']),
                            "states": [] # Empty states indicates no 3D data loaded
                        })

                res_obj = {
                    "summary": summary,
                    "frames": frames,
                    "geo": geo,
                    "cfg": cfg
                }
                results.append(res_obj)
                
        return geo, cfg, results
