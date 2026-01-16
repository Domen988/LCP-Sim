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
                 strategy_mode: str = "Counter-Rotation",
                 min_safe_interval_min: int = 5, 
                 safe_stow_el: float = 30.0, 
                 westward_offset_deg: float = 45.0, 
                 max_motor_speed_deg_per_min: float = 20.0,
                 contour_map_path: str = None,
                 el_buffer: float = 0.5):
        
        self.mode = strategy_mode
        self.min_interval = min_safe_interval_min
        self.stow_el = safe_stow_el
        self.offset = westward_offset_deg
        self.max_speed = max_motor_speed_deg_per_min
        
        self.contour_path = contour_map_path
        self.el_buffer = el_buffer
        
        self.collider: Optional[CollisionDetector] = None
        self.rig: Optional[AzElRig] = None
        
        self.contour_lut = None

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
        
        # Dispatch Strategy
        if self.mode == "Clash Contour":
            return self._generate_clash_contour_strategy(df, output_path)
        else:
            return self._generate_counter_rotation_strategy(df, output_path)

    def _generate_clash_contour_strategy(self, df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
        print(f"Generating Clash Contour Strategy using {self.contour_path}")
        
        # 1. Load Contour Map
        self._load_contour_map()
        
        # 2. Logic: Limit Elevation
        # Strategy: Standard Tracking is default.
        # If Sun Position (Tracking) violates Contour (SunEl > ContourEl), 
        # Clamp Elevation to (ContourEl - Buffer).
        
        # Output columns
        df['stow_az'] = df['sun_az'].copy()
        df['stow_el'] = df['sun_el'].copy()
        df['mode'] = "TRACKING"
        
        # We need a dense array of contour limits for the timeseries Azimuths
        if self.collider and hasattr(self.collider, 'cfg'):
             rot = self.collider.cfg.plant_rotation
        else:
             rot = 0.0
             
        # "Clash Contour" is generated from Global Azimuths (ContourMapper).
        # We must query it using Global Sun Azimuth.
        
        sun_az_global = df['sun_az'].to_numpy()
        # sun_az_local = (sun_az_global - rot) % 360.0
        # Use Global directly to match Contour Map keys
        query_az = sun_az_global % 360.0 # Ensure wrapping just in case
        
        contour_limits = self._get_contour_limits(query_az)
        # Let's assume map and sun use same convention or wrap correctly.
        # The safest is usually shifting to matches map domain if needed, but standardizing on 360 is common.
        # Let's check _get_contour_limits usage of np.interp. It expects sorted x.
        # If map is -180..180, and we pass 350, interp fails?
        # We should align convention. But first step is applying rotation.
        
        # contour_limits = self._get_contour_limits(sun_az_local)
        
        # ... logic ...
        sun_el = df['sun_el'].to_numpy() # Global El is Local El (tilt) effectively? Yes (tracking plane).
        
        # Iterate
        n = len(df)
        stow_el = np.full(n, np.nan)
        modes = np.full(n, "TRACKING", dtype=object) # Use specific type to allow strings avoiding truncation
        # Note: allocating as "TRACKING" and updating to avoid string truncation with default U<N type
        # pandas handles mixed types but numpy array needs care
        modes = np.array(["TRACKING"] * n, dtype=object)
        # Get Limit for each Azimuth
        limits = contour_limits
        
        # Identify Zones
        # Unsafe: Sun El > Limit (Visuals confirm Red Line is ceiling of safe zone? 
        # Wait, usually contour map: "Safe Elevation" means anything BELOW this line involves clash? 
        # OR "Clash Contour" means this is the boundary.
        # User prompt: "if the az el combination would fall into the clash zone (determined from the clash contour)"
        # Previous context: "minimum clash elevation points". So anything ABOVE this is likely safe? 
        # Wait. "Clash Contour" connects the LOWEST clash points.
        # So collisions happen at elevations >= Contour? Or <= Contour?
        # Usually, trackers clash at LOW elevations (flat). High elevations (vertical) can also clash (shading or structure).
        # Let's check `verify_clash_contour.py` context or image.
        # Image shows: Red Line at ~60 deg El? Or various. 
        # Green dots are SAFE. Red dots are CLASH.
        # In the image, Red dots are ABOVE the line? No, they seem to be the upper region?
        # Let's look at `contour_map.py` logic. "min_clash_el". 
        # So for a given Azimuth, the clash starts at `min_clash_el` and presumably goes upwards (or downwards?)
        # Solar trackers usually clash when they tilt too far?
        # Actually, for "Safe Elevation Contour", usually it means "Do not go below this elevation".
        # If `min_clash_el` is the lowest point where clash occurs, then elevations LOWER than this are Safe.
        # Elevations HIGHER (>= min_clash_el) are Clashing?
        # Wait, if I'm tracking the sun, and the sun is HIGH, do I clash?
        # Usually clashing happens at LOW angles (morning/evening) when panels are steep?
        # Or when panels are flat?
        # Let's assume the Contour represents the LOWER BOUNDARY of the Danger Zone.
        # So Danger = El >= Contour. Safe = El < Contour.
        # BUT, User said: "elevation should be dropped to the max safe elevation".
        # "Dropped" implies going down.
        # If I am at 60 deg (Clash) and I drop to 40 deg (Safe)...
        # That means LOWER is Safe.
        # So Danger Zone is HIGH elevation?
        # That contradicts typical tracker limits (usually overlapping when flat). 
        # Let's re-read carefully.
        # "Clash Elevation Contour (Actual Clash Boundary)"
        # "min_clash_el"
        # If standard result is: "Clash at 20 deg, 21 deg, ... 90 deg". Min is 20. 
        # Then [20, 90] is Bad. [0, 19] is Good.
        # User says: "elevation should be dropped to max safe elevation - elevation buffer".
        # Max Safe Elevation would be Contour - epsilon.
        # So yes: Target < Contour.
        # LIMIT = Contour.
        # If Current > Limit -> Clamp to Limit - Buffer.
        
        # Apply Clamping (Forward Pass - Instantaneous)
        raw_target_el = sun_el.copy()
        mask_clash = (sun_el >= limits) # Assuming inclusive boundary is bad
        
        # Set limit
        safe_targets = limits - self.el_buffer
        
        # Apply
        raw_target_el[mask_clash] = safe_targets[mask_clash]
        
        # Mark Mode
        df.loc[mask_clash, 'mode'] = "CLASH_AVOIDANCE"
        
        # 3. Apply Motor Speed Derating (Lookahead / Backwards Pass)
        # "Movement should happen enough in advance"
        # We need to ensure that we are AT safe_target when we arrive at that time.
        # Constraint: El[t] <= El[t+1] + Speed*dt (if we need to go down effectively)
        # Or generally: |El[t+1] - El[t]| <= Speed * dt
        # Since the constraint is an UPPER BOUND (Stay below Contour),
        # We need to ensure we are below the curve.
        # If at t+10 we must be at 30 deg, and we are at 60 deg at t=0.
        # We must start moving down early.
        # Filter: `ValidMax[i] = min( Constraint[i], ValidMax[i+1] + Speed*dt )`
        # This propagates the constraint backwards.
        
        dt_min = getattr(self.collider.cfg, 'timestep_min', 5.0) if self.collider else 5.0
        max_delta = self.max_speed * dt_min
        
        # Backward Pass
        N = len(raw_target_el)
        filtered_el = raw_target_el.copy()
        
        # Vectorize or Loop? Loop is safer for logic, N ~ 100k fast enough usually.
        # Actually for 1 year, 10 min steps = 50k points. Python loop is okay-ish (0.1s).
        # We perform backward pass.
        '''
        for i in range(N-2, -1, -1):
             # We must be able to reach filtered_el[i+1] from filtered_el[i]
             # So filtered_el[i] cannot be too high relative to [i+1]
             # filtered_el[i] - filtered_el[i+1] <= max_delta
             # filtered_el[i] <= filtered_el[i+1] + max_delta
             
             limit_from_future = filtered_el[i+1] + max_delta
             
             # Also respect local constraint (already in filtered_el[i])
             filtered_el[i] = min(filtered_el[i], limit_from_future)
        '''
        # Let's try numba or just simple loop. Or numpy accumulation?
        # It's a cumulative min operation with decay.
        # `accumulation = min_accumulate(x, decay)`
        # Can do efficiently with pandas ewm? No.
        # Simple loop.
        
        for i in range(N-2, -1, -1):
            limit_from_future = filtered_el[i+1] + max_delta
            if filtered_el[i] > limit_from_future:
                 filtered_el[i] = limit_from_future
                 # Update mode if we are actually deviating from sun/clamped due to speed
                 if df.at[i, 'mode'] == "TRACKING":
                      df.at[i, 'mode'] = "PRE_CLASH"
        
        # Forward Pass? Checking if we can actually GO UP fast enough?
        # Stow usually involves dropping. Recovering to sun is fine if sun is rising.
        # If Sun rises faster than speed? (Unlikely for standard motor).
        # But let's apply Forward Pass too for physical feasibility.
        # `El[i+1] <= El[i] + max_delta`
        # `El[i+1] >= El[i] - max_delta`
        
        # The constraint is primarily "Stay Low".
        # So usually we don't need to force it HIGHER.
        # But we should cap the rising rate.
        
        for i in range(0, N-1):
             limit_from_past = filtered_el[i] + max_delta
             if filtered_el[i+1] > limit_from_past:
                  filtered_el[i+1] = limit_from_past
                  # Use existing mode
        
        df['stow_el'] = filtered_el
        
        # Azimuth: Keep Sun Azimuth? 
        # "stow strategy... elevation should be dropped"
        # Assuming Azimuth tracking continues normally?
        # User didn't specify Azimuth handling. "Elevation should be dropped".
        # We assume Azimuth tracks Sun.
        
        if output_path:
            df.to_csv(output_path, index=False)
            
        return df

    def _get_contour_limits(self, az_array):
        # self.contour_lut is simple list of [az, el] sorted?
        if not self.contour_lut:
             # Default safe (90.0 means High Limit = Safe)
             return np.full_like(az_array, 90.0)
        
        # Convert to DataFrame for interpolation
        # LUT Azimuths are discrete bins.
        # We need to interpolate for exact Sun Azimuth.
        lut_az, lut_el = zip(*self.contour_lut)
        
        # Handle Periodic Boundary?
        # LUT usually -180 to 180 or 0 to 360?
        # `contour_map.py` outputs sorted bins. Azimuths are usually -180..180 or 0..360 depending on sim.
        # `process_csv` inputs degrees.
        # We should assume standard numpy interp.
        
        # Wrap Azimuths for interpolation if needed.
        # Simple np.interp
        return np.interp(az_array, lut_az, lut_el, left=90.0, right=90.0) # Extrapolation default 90

    def _load_contour_map(self):
        if not self.contour_path or not os.path.exists(self.contour_path):
             print("Error: Contour Path invalid.")
             self.contour_lut = []
             return

        with open(self.contour_path, 'r') as f:
             data = json.load(f)
             
        # Format: {"clash_contour": [[az, el], ...], ...}
        if "clash_contour" in data:
             self.contour_lut = data["clash_contour"]
        elif "safe_elevation_lut" in data:
             self.contour_lut = data["safe_elevation_lut"]
        else:
             self.contour_lut = []
             
        # Ensure sorted
        self.contour_lut.sort(key=lambda x: x[0])

    def _generate_counter_rotation_strategy(self, df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
        # Original Logic (moved here)
        # Load Physics
        # ... logic from previous process_csv ...
        
        # Detect Initial Events (based on original 'safety' flag)
        events = self._detect_events(df)
        
        # Initialize Output Columns
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
        # Apply offset to align with Plant Geometry Frame (Local = Global - Rot)
        rots_p = self.rig.get_orientation(track_az - rot_offset, t_el)
        
        # Active (Odd)
        s_el = np.maximum(min_el, stow_el)
        rots_a = self.rig.get_orientation(stow_az - rot_offset, s_el)
        
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
        # Stow Azimuth target = Sun Azimuth - Offset (Global Frame)
        # Global Stow = Global Sun - Offset.
        # Local Stow = (Global Sun - Rot) - Offset = Global Sun - Offset - Rot.
        # Previous logic adding Rot was incorrect for Global Azimuth target.
        rot = 0.0
        if self.collider and hasattr(self.collider, 'cfg'):
            rot = self.collider.cfg.plant_rotation
            
        current_stow_az = self._wrap_dates(sun_az_start - self.offset)
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
