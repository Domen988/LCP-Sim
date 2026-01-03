
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

    def list_simulations_details(self) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts containing metadata for all available simulations.
        Used for the detailed Load Dialog.
        """
        if not os.path.exists(self.base_path):
            return []
        
        sims_details = []
        for name in os.listdir(self.base_path):
            path = os.path.join(self.base_path, name)
            cfg_path = os.path.join(path, "config.json")
            
            if os.path.isdir(path) and os.path.exists(cfg_path):
                details = {"name": name}
                try:
                    with open(cfg_path, 'r') as f:
                        data = json.load(f)
                        
                    # Extract valuable metadata
                    cfg = data.get('config', {})
                    geo = data.get('geometry', {})
                    stow = data.get('stow_strategy', None)
                    
                    details['timestamp'] = data.get('timestamp', "")
                    details['panels'] = cfg.get('total_panels', 0)
                    details['duration'] = cfg.get('duration_days', 0)
                    details['sun_source'] = cfg.get('sun_source', 'pvlib')
                    
                    # Extended Metadata
                    details['tolerance'] = cfg.get('tolerance', 0.0)
                    details['rotation'] = cfg.get('plant_rotation', 0.0)
                    details['timestep'] = cfg.get('timestep_min', 0)
                    
                    if stow:
                        details['type'] = "Stow Strategy"
                        details['source_sim'] = stow.get('source_simulation', 'Unknown')
                        details['stow_gap'] = stow.get('min_safe_interval_min', '')
                        details['stow_el'] = stow.get('safe_stow_el', '')
                        details['stow_offset'] = stow.get('westward_offset_deg', '')
                        details['stow_speed'] = stow.get('max_motor_speed_deg_per_min', '')
                    else:
                        details['type'] = "Standard"
                        details['source_sim'] = "-"
                        details['stow_gap'] = "-"
                        details['stow_el'] = "-"
                        details['stow_offset'] = "-"
                        details['stow_speed'] = "-"
                        
                except Exception as e:
                    print(f"Error reading config for {name}: {e}")
                    details['error'] = str(e)
                    
                sims_details.append(details)
                
        # Sort by timestamp desc (newest first)
        try:
            sims_details.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        except:
            pass
            
        return sims_details

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

    def save_simulation(self, name: str, geo: PanelGeometry, cfg: ScenarioConfig, results: List[Dict[str, Any]], extra_data: Dict[str, Any] = None) -> str:
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
        geo_dict = dataclasses.asdict(geo)
        
        cfg_dict = dataclasses.asdict(cfg)

        # A. Inject Pivot Depth (User Input)
        # pivot_depth = offset_z + thickness/2
        offset_z = geo_dict.get('pivot_offset', [0,0,0])[2]
        thickness = geo_dict.get('thickness', 0.15)
        geo_dict['pivot_depth'] = offset_z + (thickness / 2.0)
        
        # B. Inject Simulation Runtime Params (Timestep, Duration)
        # Duration
        cfg_dict['duration_days'] = len(results)
        
        # Timestep (Infer from first valid pair of frames)
        timestep_min = 0
        try:
            for day_res in results:
                frames = day_res.get('frames', [])
                if len(frames) >= 2:
                    t1 = frames[0]['time']
                    t2 = frames[1]['time']
                    if hasattr(t1, 'to_pydatetime'): t1 = t1.to_pydatetime()
                    if hasattr(t2, 'to_pydatetime'): t2 = t2.to_pydatetime()
                    
                    diff = t2 - t1
                    timestep_min = int(diff.total_seconds() / 60)
                    break 
        except Exception:
            pass
        
        cfg_dict['timestep_min'] = timestep_min

        data = {
            "geometry": geo_dict,
            "config": cfg_dict,
            "version": "1.1",
            "timestamp": datetime.now().isoformat()
        }
        
        if extra_data:
            data.update(extra_data)
        
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
            
        # Strip extra fields not in dataclass (Backward compat for new files loading into old classes)
        geo_data.pop('pivot_depth', None)
        
        geo = PanelGeometry(**geo_data)
        
        # Config
        cfg_data = data.get("config", {})
        # Strip derived inputs
        cfg_data.pop('timestep_min', None)
        cfg_data.pop('duration_days', None)
        
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
                        frame_dict = {
                            "time": ts_row['time'].to_pydatetime(),
                            "sun_az": float(ts_row['sun_az']),
                            "sun_el": float(ts_row['sun_el']),
                            "safety": bool(ts_row['safety']),
                            "act_w": float(ts_row['act_w']),
                            "theo_w": float(ts_row['theo_w']),
                            "states": [] # Empty states indicates no 3D data loaded
                        }
                        
                        # Load Stow Columns if present
                        try:
                            if 'stow_az' in ts_row and pd.notna(ts_row['stow_az']):
                                 frame_dict['stow_az'] = float(ts_row['stow_az'])
                            if 'stow_el' in ts_row and pd.notna(ts_row['stow_el']):
                                 frame_dict['stow_el'] = float(ts_row['stow_el'])
                        except Exception:
                            pass
                             
                        frames.append(frame_dict)

                res_obj = {
                    "summary": summary,
                    "frames": frames,
                    "geo": geo,
                    "cfg": cfg
                }
                results.append(res_obj)
                
        return geo, cfg, results
        
    def delete_simulation(self, name: str) -> bool:
        """
        Deletes a simulation directory.
        Includes handling for Windows 'Access Denied' on read-only files.
        """
        import stat
        
        sim_dir = os.path.join(self.base_path, name)
        
        # Guard against deleting current working directory
        try:
            if os.path.abspath(sim_dir) == os.getcwd():
                os.chdir(os.path.dirname(os.path.abspath(sim_dir)))
        except:
            pass

        if os.path.exists(sim_dir) and os.path.isdir(sim_dir):
            def on_rm_error(func, path, exc_info):
                # Attempt to fix read-only files/folders
                os.chmod(path, stat.S_IWRITE)
                try:
                    func(path)
                except Exception:
                    pass
                    
            try:
                shutil.rmtree(sim_dir, onerror=on_rm_error)
                return True
            except Exception as e:
                print(f"Error deleting simulation: {e}")
                return False
        return False
