import pandas as pd
import numpy as np
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

from lcp.core.geometry import PanelGeometry
from lcp.core.config import ScenarioConfig
from lcp.physics.collisions import CollisionDetector
from lcp.physics.kinematics import AzElRig

@dataclass
class ClashEvent:
    start_idx: int
    end_idx: int
    type: str = "clash" # 'clash' or 'merged'

class StowStrategyGenerator:
    """
    Generates safe stow profiles for clash events in simulation data.
    """
    
    def export_strategy_csv(self, strategy_path: str, output_path: str, sun_csv_dir: str):
        """
        Exports strategy merged with source Sun CSVs to requested format.
        Columns: Original columns + 'Stow Elevation DEG', 'Stow Azimuth DEG', 'Stow mode'
        """
        print(f"Exporting Strategy to {output_path}...")
        
        # 1. Load Strategy
        # Strategy has: time, stow_az, stow_el, mode, safety, sun_az, sun_el
        df_strat = pd.read_csv(strategy_path)
        # Standardize column name just in case
        if 'Timestamp' in df_strat.columns:
             df_strat.rename(columns={'Timestamp': 'time'}, inplace=True)
             
        df_strat['time'] = pd.to_datetime(df_strat['time'])
        df_strat.sort_values('time', inplace=True)
        
        # 2. Iterate Strategy Time Range (Year/Month)
        # We process month by month to load relevant source files
        # 2. Iterate Strategy Time Range (Year/Month)
        # We process month by month to load relevant source files
        start_date = df_strat['time'].min()
        end_date = df_strat['time'].max()
        
        # Generate list of (Year, Month) covering the range
        dates = pd.date_range(start_date, end_date, freq='MS')
        if len(dates) == 0:
            # Single month or less
             dates = pd.DatetimeIndex([start_date])
             
        full_dfs = []
        
        processed_months = set()
        
        for d in dates:
            yr = d.year
            mo = d.month
            if (yr, mo) in processed_months: continue
            processed_months.add((yr, mo))
            
            # Load Source CSV
            src_name = f"{yr}_{mo:02d}.csv"
            src_path = os.path.join(sun_csv_dir, src_name)
            
            if not os.path.exists(src_path):
                print(f"Warning: Source CSV {src_name} missing for export. Skipping period.")
                continue
                
            # Load Source
            # Header format: JULIAN DAY UTC,YEAR,Month ,Day,Hours Decimal,Time Sexa Minutes,Time UTC,Azimuth DEG,Elevation DEG,Earth Declination DEG,Equation of Time MINUTES
            df_src = pd.read_csv(src_path)
            # Clean cols
            col_map = {c: c.strip() for c in df_src.columns}
            df_src.rename(columns=col_map, inplace=True)
            
            # Construct DateTime index for Source to allow alignment
            # (Similar to SunProvider loading)
            df_src['__dt__'] = pd.to_datetime(
                df_src[['YEAR', 'Month', 'Day']].astype(str).agg('-'.join, axis=1) + ' ' + df_src['Time UTC']
            )
            
            # Filter Strategy to this month
            mask = (df_strat['time'].dt.year == yr) & (df_strat['time'].dt.month == mo)
            df_strat_mo = df_strat[mask].copy()
            
            if df_strat_mo.empty:
                continue
            
            # Merge
            # Left join on Source
            merged = pd.merge(
                df_src, 
                df_strat_mo[['time', 'stow_az', 'stow_el', 'mode']], 
                left_on='__dt__', 
                right_on='time', 
                how='left'
            )
            
            # Set Index for interpolation
            merged.set_index('__dt__', inplace=True)
            
            # Interpolate Numeric
            merged['Stow Elevation DEG'] = merged['stow_el'].interpolate(method='time', limit_direction='both')
            merged['Stow Azimuth DEG'] = merged['stow_az'].interpolate(method='time', limit_direction='both')
            
            # Fill Categorical (Mode) - Forward fill
            merged['Stow mode'] = merged['mode'].ffill().bfill() 
            
            # Drop join columns
            merged.drop(columns=['time', 'stow_az', 'stow_el', 'mode'], inplace=True)
            
            # Append to list
            full_dfs.append(merged)
            
        if not full_dfs:
            print("No matching source data found to export.")
            return

        # Concat
        final_df = pd.concat(full_dfs)
        
        # Select/Order Columns match Source + New
        # Source Cols (minus __dt__)
        base_cols = [c for c in df_src.columns if c != '__dt__']
        new_cols = ['Stow Elevation DEG', 'Stow Azimuth DEG', 'Stow mode']
        
        final_cols = base_cols + new_cols
        
        # Filter (df_src was modified in loop? No, fresh load)
        # But `merged` has all cols.
        final_df = final_df[final_cols]
        
        # Save
        final_df.to_csv(output_path, index=False)
        print(f"Export Complete: {output_path}")
        
    def __init__(self, 
                 min_safe_interval_min: int = 5, 
                 safe_stow_el: float = 30.0, 
                 westward_offset_deg: float = 45.0, 
                 max_motor_speed_deg_per_min: float = 20.0):
        
        self.min_interval = min_safe_interval_min
        self.stow_el = safe_stow_el
        self.offset = westward_offset_deg
        self.max_speed = max_motor_speed_deg_per_min
        
        self.collider: Optional[CollisionDetector] = None
        self.rig: Optional[AzElRig] = None
        
    def process_csv(self, file_path: str, config_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Loads CSV, processes kinematic strategy iteratively until safe, and returns augmented DataFrame.
        """
        # Load Data
        df = pd.read_csv(file_path)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])

        # Load Physics
        self._load_physics(config_path)

        # Detect Initial Events (based on original 'safety' flag)
        events = self._detect_events(df)
        
        # Initialize Output Columns
        # We start with Tracking assumed (Sun Pos) to fill gaps
        # But we'll overwrite tracking with "Standard Tracking" logic or just keep NaNs where not stow?
        # The prompt implies we should output a file that has stow_az/el where stowing, 
        # and standard tracking otherwise.
        
        # Let's Init with NaNs, fill tracking later
        df['stow_az'] = np.nan
        df['stow_el'] = np.nan
        df['mode'] = "TRACKING"

        # Iteration Loop
        max_iterations = 20
        iteration = 0
        
        while iteration < max_iterations:
            print(f"Iteration {iteration + 1}: Generating Strategy for {len(events)} events...")
            
            # 1. Generate Strategy (Core + Ramps)
            df['stow_az'] = np.nan
            df['stow_el'] = np.nan
            df['mode'] = "TRACKING"
            
            for ev in events:
                self._generate_core_maneuver(df, ev)
                
            for ev in events:
                self._solve_ramps(df, ev)
                
            # Fill Tracking for Safety Check
            # stowing_profile (Active)
            check_stow_az = df['stow_az'].fillna(df['sun_az']).to_numpy()
            check_stow_el = df['stow_el'].fillna(df['sun_el']).to_numpy()
            
            # tracking_profile (Passive)
            check_track_az = df['sun_az'].to_numpy()
            check_track_el = df['sun_el'].to_numpy()
            
            # 2. Check Safety
            # Check Active Profile vs Passive Profile (Checkerboard Interaction)
            clash_mask = self._check_conflicts(check_track_az, check_track_el, check_stow_az, check_stow_el)
            
            new_clashes = np.where(clash_mask)[0]
            if len(new_clashes) == 0:
                print("Safety Check Passed!")
                break
                
            print(f"Found {len(new_clashes)} unsafe timestamps. Expanding events...")
            
            # Debug Clash Modes
            modes = df.loc[new_clashes, 'mode'].unique()
            print(f"Clash Modes detected: {modes}")
            
            # 3. Expand Events
            # Create a synthetic 'safety' series combining original + new clashes
            event_mask = np.zeros(len(df), dtype=bool)
            for ev in events:
                event_mask[ev.start_idx : ev.end_idx + 1] = True
            
            # Add new clashes with PADDING to force wider safety margin
            # Directional Padding based on Mode
            padded_clash_mask = clash_mask.copy()
            if len(new_clashes) > 0:
                 for i in new_clashes:
                     mode = df.loc[i, 'mode']
                     padded_clash_mask[i] = True
                     
                     if mode == 'RAMP_EXIT':
                         # Extend stow later (Right)
                         if i < len(df) - 1: padded_clash_mask[i+1] = True
                     elif mode == 'RAMP_ENTRY':
                         # Extend stow earlier (Left)
                         if i > 0: padded_clash_mask[i-1] = True
                     else:
                         # Core or Tracking - Padding both sides safe default
                         if i > 0: padded_clash_mask[i-1] = True
                         if i < len(df) - 1: padded_clash_mask[i+1] = True
            
            combined_mask = event_mask | padded_clash_mask
            
            # Re-detect and Merge using _detect_events_from_mask logic
            new_events = self._detect_events_from_mask(df, combined_mask)
            
            # Convergence Check
            if len(new_events) == len(events) and np.array_equal(combined_mask, event_mask):
                print(f"Warning: Iteration {iteration} did not change event structure. Stopping to avoid infinite loop.")
                print(f"Indices of persistent clashes: {new_clashes}")
                print(f"Modes: {df.loc[new_clashes, 'mode'].values}")
                break
                
            events = new_events
            iteration += 1

        # Final Fill
        mask_track = df['stow_az'].isna()
        df.loc[mask_track, 'stow_az'] = df.loc[mask_track, 'sun_az']
        df.loc[mask_track, 'stow_el'] = df.loc[mask_track, 'sun_el']
        
        if output_path:
            df.to_csv(output_path, index=False)
            
        return df

    def _load_physics(self, config_path: str):
        with open(config_path, "r") as f:
            data = json.load(f)
            
        # Handle Load logic similar to persistence
        geo_data = data.get("geometry", {})
        if "pivot_offset" in geo_data and isinstance(geo_data["pivot_offset"], list):
            geo_data["pivot_offset"] = tuple(geo_data["pivot_offset"])
        geo_data.pop('pivot_depth', None)
        geo = PanelGeometry(**geo_data)
        
        cfg_data = data.get("config", {})
        cfg_data.pop('timestep_min', None)
        cfg_data.pop('duration_days', None)
        cfg = ScenarioConfig(**cfg_data)
        
        self.collider = CollisionDetector(geo, cfg)
        self.rig = AzElRig()

    def _check_conflicts(self, track_az: np.ndarray, track_el: np.ndarray, 
                         stow_az: np.ndarray, stow_el: np.ndarray) -> np.ndarray:
        """
        Returns boolean array where True = Collision.
        Checks Passive (Tracking) vs Active (Stow Strategy) neighbors (Checkerboard).
        """
        if self.collider is None or self.rig is None:
            raise RuntimeError("Physics not initialized")
            
        # Vectorized Check
        min_el = getattr(self.collider.cfg, 'min_elevation', 15.0)
        
        # Apply Plant Rotation Correction
        # Replay Logic: local_az = sun_az - (-plant_rotation)
        rot_offset = self.collider.cfg.plant_rotation
        
        # 1. Get Rotations
        # Passive (Even)
        t_el = np.maximum(min_el, track_el)
        # Apply offset to align with Plant Geometry Frame
        rots_p = self.rig.get_orientation(track_az + rot_offset, t_el)
        
        # Active (Odd)
        s_el = np.maximum(min_el, stow_el)
        rots_a = self.rig.get_orientation(stow_az + rot_offset, s_el)
        
        N = len(track_az)
        clash_mask = np.zeros(N, dtype=bool)
        
        px = self.collider.cfg.grid_pitch_x
        py = self.collider.cfg.grid_pitch_y
        
        pivot_00 = np.array([0.0, 0.0, 0.0])
        
        # Checkerboard Logic:
        # (0,0) is Passive.
        # Check Orthogonal Neighbors (Active): E, W, N, S.
        # Check Diagonal Neighbors (Passive): NE, SE, SW, NW (Self Check - usually safe if valid tracking)
        # But for "Stow Strategy" validation, the critical interaction is Passive vs Active.
        
        # Orthogonal Neighbors (Distance to Active Panels)
        neighbors_cross = [
            np.array([px, 0.0, 0.0]),    # E
            np.array([-px, 0.0, 0.0]),   # W
            np.array([0.0, py, 0.0]),    # S
            np.array([0.0, -py, 0.0]),   # N
        ]
        
        for nb_pos in neighbors_cross:
            # Check Passive(0,0) vs Active(Neighbor)
            c = self.collider.check_clash(pivot_00, nb_pos, rots_p, rot_b=rots_a) 
            clash_mask |= c
            
        return clash_mask

    def _detect_events(self, df: pd.DataFrame) -> List[ClashEvent]:
        # Handle boolean or string parsing
        if df['safety'].dtype == 'O':
            is_clash = df['safety'].astype(str).str.lower() == 'true'
        else:
            is_clash = df['safety'].astype(bool)
        
        return self._detect_events_from_mask(df, is_clash)

    def _detect_events_from_mask(self, df: pd.DataFrame, mask: np.ndarray) -> List[ClashEvent]:
        clash_indices = np.where(mask)[0]
        
        if len(clash_indices) == 0:
            return []
            
        # 1. Group contiguous
        raw_events = []
        start = clash_indices[0]
        end = clash_indices[0]
        
        for i in range(1, len(clash_indices)):
            idx = clash_indices[i]
            if idx == end + 1:
                end = idx
            else:
                raw_events.append(ClashEvent(start, end))
                start = idx
                end = idx
        raw_events.append(ClashEvent(start, end))
        
        # 2. Merge Gap Logic
        merged_events = []
        current_ev = raw_events[0]
        
        timestamps = df['time']
        
        for next_ev in raw_events[1:]:
            t_end = timestamps[current_ev.end_idx]
            t_start = timestamps[next_ev.start_idx]
            
            gap_min = (t_start - t_end).total_seconds() / 60.0
            
            if gap_min <= self.min_interval:
                current_ev.end_idx = next_ev.end_idx
                current_ev.type = "merged"
            else:
                merged_events.append(current_ev)
                current_ev = next_ev
                
        merged_events.append(current_ev)
        return merged_events

    def _generate_core_maneuver(self, df: pd.DataFrame, ev: ClashEvent):
        """
        Calculates 1:1 Counter-Rotation inside the event window.
        """
        idxs = range(ev.start_idx, ev.end_idx + 1)
        
        df.loc[idxs, 'stow_el'] = self.stow_el
        df.loc[idxs, 'mode'] = "STOW"
        
        # Initialize Start
        t_start_idx = ev.start_idx
        sun_az_start = df.loc[t_start_idx, 'sun_az']
        
        # Apply offset correction for plant rotation
        # Stow Azimuth target = Sun Azimuth - Offset (Relative to Plant?)
        # User requested: "stow westward offset is also not taking into account the plant rotation"
        # We add rotation to align the "West" logic with the plant frame.
        rot = 0.0
        if self.collider and hasattr(self.collider, 'cfg'):
            rot = self.collider.cfg.plant_rotation
            
        current_stow_az = self._wrap_dates(sun_az_start - self.offset + rot)
        df.loc[t_start_idx, 'stow_az'] = current_stow_az
        
        # Iterative update
        for i in range(ev.start_idx + 1, ev.end_idx + 1):
            prev_idx = i - 1
            sun_curr = df.loc[i, 'sun_az']
            sun_prev = df.loc[prev_idx, 'sun_az']
            
            sun_delta = self._diff_angle(sun_curr, sun_prev)
            panel_delta = -sun_delta # Counter-Rotate
            
            prev_stow = df.loc[prev_idx, 'stow_az']
            new_stow = self._wrap_dates(prev_stow + panel_delta)
            
            df.loc[i, 'stow_az'] = new_stow

    def _solve_ramps(self, df: pd.DataFrame, ev: ClashEvent) -> bool:
        """
        Generates Entry and Exit Ramps.
        Expands iteratively if velocity violation occurs.
        """
        timestamps = df['time']
        
        # --- ENTRY RAMP ---
        target_idx = ev.start_idx
        # If target already invalid (e.g. at 0), skip
        if target_idx <= 0: return False

        valid_ramp = False
        ramp_start_idx = target_idx - 1
        
        target_az = df.loc[target_idx, 'stow_az']
        target_el = df.loc[target_idx, 'stow_el']
        
        while not valid_ramp and ramp_start_idx >= 0:
            start_t = timestamps[ramp_start_idx]
            end_t = timestamps[target_idx]
            duration_min = (end_t - start_t).total_seconds() / 60.0
            
            if duration_min <= 0: break
            
            # Start State (Tracking)
            # Use computed stow_az if available (overlap), else Sun
            if pd.notna(df.loc[ramp_start_idx, 'stow_az']):
                 # Overlap with previous event!
                 # In this loop logic, we shouldn't simple expand into another event?
                 # But merge logic should have handled this. 
                 # If we hit an existing stow, we just splice?
                 start_az = df.loc[ramp_start_idx, 'stow_az']
                 start_el = df.loc[ramp_start_idx, 'stow_el']
            else:
                 start_az = df.loc[ramp_start_idx, 'sun_az']
                 start_el = df.loc[ramp_start_idx, 'sun_el']
            
            d_az = abs(self._diff_angle(target_az, start_az))
            d_el = abs(target_el - start_el)
            
            v_az = d_az / duration_min
            v_el = d_el / duration_min
            
            if v_az <= self.max_speed and v_el <= self.max_speed:
                self._interpolate_ramp(df, ramp_start_idx, target_idx, 
                                       start_az, start_el, target_az, target_el, 
                                       "RAMP_ENTRY")
                valid_ramp = True
            else:
                ramp_start_idx -= 1
                
        # --- EXIT RAMP ---
        source_idx = ev.end_idx
        if source_idx >= len(df) - 1: return False
        
        valid_exit = False
        ramp_end_idx = source_idx + 1
        
        source_az = df.loc[source_idx, 'stow_az']
        source_el = df.loc[source_idx, 'stow_el']
        
        while not valid_exit and ramp_end_idx < len(df):
            start_t = timestamps[source_idx]
            end_t = timestamps[ramp_end_idx]
            duration_min = (end_t - start_t).total_seconds() / 60.0
            
            if duration_min <= 0: break
            
            # Target (Tracking or Next Event?)
            if pd.notna(df.loc[ramp_end_idx, 'stow_az']):
                target_az = df.loc[ramp_end_idx, 'stow_az']
                target_el = df.loc[ramp_end_idx, 'stow_el']
            else:
                target_az = df.loc[ramp_end_idx, 'sun_az']
                target_el = df.loc[ramp_end_idx, 'sun_el']
            
            d_az = abs(self._diff_angle(target_az, source_az))
            d_el = abs(target_el - source_el)
            
            v_az = d_az / duration_min
            v_el = d_el / duration_min
            
            if v_az <= self.max_speed and v_el <= self.max_speed:
                self._interpolate_ramp(df, source_idx, ramp_end_idx, 
                                       source_az, source_el, target_az, target_el, 
                                       "RAMP_EXIT")
                valid_exit = True
            else:
                ramp_end_idx += 1
                
        return valid_ramp and valid_exit

    def _interpolate_ramp(self, df, start_idx, end_idx, s_az, s_el, e_az, e_el, mode):
        steps = end_idx - start_idx
        if steps <= 0: return
        
        delta_az = self._diff_angle(e_az, s_az) 
        delta_el = e_el - s_el
        
        for i in range(1, steps): 
            frac = i / float(steps)
            cur_idx = start_idx + i
            
            interp_az = self._wrap_dates(s_az + delta_az * frac)
            interp_el = s_el + delta_el * frac
            
            # Don't overwrite if it's already 'STOW' (Core)? 
            # Ramps shouldn't overlap Core. Core has priority.
            if df.loc[cur_idx, 'mode'] == 'STOW':
                continue
                
            df.loc[cur_idx, 'stow_az'] = interp_az
            df.loc[cur_idx, 'stow_el'] = interp_el
            df.loc[cur_idx, 'mode'] = mode

    def _wrap_dates(self, angle):
        return angle % 360.0

    def _diff_angle(self, target, source):
        diff = (target - source + 180) % 360 - 180
        return diff
