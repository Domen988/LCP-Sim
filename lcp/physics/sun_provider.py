
import pandas as pd
import numpy as np
import os
import glob
from typing import Tuple, Optional
from datetime import datetime

class SunPositionProvider:
    """
    Provides sun position data (Azimuth, Elevation) from configured source.
    """
    
    def __init__(self, source_mode: str = 'pvlib', csv_dir: str = None):
        self.source_mode = source_mode
        self.csv_dir = csv_dir
        self.cached_csv_data = {} # (year, month) -> dataframe
        
    def get_sun_positions(self, 
                          times: pd.DatetimeIndex, 
                          solar_engine) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (azimuth, elevation) arrays for the given timestamps.
        """
        if self.source_mode == 'csv' and self.csv_dir:
            try:
                return self._get_from_csv(times, solar_engine)
            except Exception as e:
                print(f"Error loading sun from CSV: {e}. Falling back to PVLib.")
                return self._get_from_pvlib(times, solar_engine)
        else:
            return self._get_from_pvlib(times, solar_engine)
            
    def _get_from_pvlib(self, times, solar_engine) -> Tuple[np.ndarray, np.ndarray]:
        # Use existing solar engine logic (PVLib)
        # Note: solar_engine usually takes single time or list. 
        # Ideally we batch this. 
        # But SimulationRunner loop usually calls this per timestep or batch?
        # SimulationRunner calls: solar.get_sun_position(time)
        
        # If we are here, we might want to batch it using pvlib directly 
        # if solar_engine exposes it, or just loop.
        # But wait, SolarEngine.calculate_position returns Az, El.
        
        # Let's assume we call this once per simulation or per day?
        # If per-timestamp:
        azs = []
        els = []
        # Batch call optimization if solar_engine supports it (it does)
        # But we need to handle return type.
        # solar_engine.get_position(times) returns DataFrame if times is list-like
        
        pos = solar_engine.get_position(times)
        # It handles timezone localization internally
        if isinstance(pos, pd.DataFrame):
            return pos['azimuth'].values, pos['elevation'].values
        else:
            # Fallback for single item (shouldn't happen with times list)
            return np.array([pos.azimuth]), np.array([pos.elevation])

    def _get_from_csv(self, times, solar_engine) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads from YYYY_MM.csv files.
        File Format:
        YEAR, Month , Day, Time UTC, Azimuth DEG, Elevation DEG
        """
        # We need to load relevant CSVs for the given times
        # Group by Year-Month
        
        azs = np.full(len(times), np.nan)
        els = np.full(len(times), np.nan)
        
        # Helper to index times
        df_req = pd.DataFrame({'time': times})
        df_req['idx'] = range(len(times))
        df_req['year'] = df_req['time'].dt.year
        df_req['month'] = df_req['time'].dt.month
        
        groups = df_req.groupby(['year', 'month'])
        
        files_loaded = 0
        
        for (year, month), group in groups:
            # 1. Check/Load CSV
            key = (year, month)
            df_src = self._load_csv_file(year, month)
            
            if df_src is None:
                # Fallback handled by caller or partial fallback here?
                # Prompt says: "some months for 2025 are missing - take ... from pvlib, but notify user"
                print(f"Warning: Sun CSV missing for {year}-{month:02d}. Using PVLib fallback.")
                # Calculate PVLib for this group
                t_group = group['time']
                az_fb, el_fb = self._get_from_pvlib(t_group, solar_engine)
                
                # Check shapes. group['idx'] is indexer into azs. 
                # az_fb is array matching t_group length.
                azs[group['idx']] = az_fb
                els[group['idx']] = el_fb
                continue
                
            # 2. Intersect/Lookup
            # Source CSV has 'Time UTC' as string "00:00". Need to combine with Year/Month/Day
            # To be efficient, let's index source by datetime?
            # Or just 'Day' + 'Time UTC' matching?
            
            # The requested times might not align perfectly with CSV minutes?
            # CSV has 1 min resolution? Sample: 00:00, 00:01...
            # We should probably interpolate or nearest match.
            
            # Reindex Source to datetime index
            # This is cached in _load_csv_file to avoid re-parsing
            
            # Find closest matches
            # Using searchsorted or merge_asof
            
            # Filter source to relevant range
            # Actually, doing merge_asof on the whole group is robust
            
            df_group_times = group[['time']].sort_values('time')
            
            # Merge
            merged = pd.merge_asof(
                df_group_times, 
                df_src, 
                left_on='time', 
                right_index=True, 
                direction='nearest', 
                tolerance=pd.Timedelta('5min') # Tolerance logic?
            )
            
            # Assign
            # Map back to original indices
            # merged has 'Azimuth DEG' and 'Elevation DEG'
            
            # We need to map `merged` values back to `azs` array using `group['idx']`
            # group was sliced from df_req. 
            # We need to be careful with ordering.
            
            # Easier:
            # 1. Get subset of indexes from group
            global_indices = group['idx'].values
            
            # 2. Get corresponding times
            t_subset = group['time'].values
            
            # 3. Lookup in df_src (Index is Datetime)
            # Use reindex with nearest?
            # df_src index is UTC time.
            
            # Convert t_subset to UTC if not already?
            # times is DatetimeIndex. 
            
            # Using index.get_indexer(method='nearest')
            indexer = df_src.index.get_indexer(t_subset, method='nearest')
            
            # Check validity (indexer != -1)
            valid = indexer != -1
            
            # Get values
            found_azs = df_src['Azimuth DEG'].iloc[indexer].values
            found_els = df_src['Elevation DEG'].iloc[indexer].values
            
            # Handle tolerance manually? Or just trust nearest.
            # If large gap, might be wrong. But CSV is dense (1 min).
            
            azs[global_indices] = found_azs
            els[global_indices] = found_els
            
            files_loaded += 1
            
        print(f"Loaded sun positions from CSV files ({files_loaded} files found).")
        return azs, els

    def _load_csv_file(self, year, month) -> Optional[pd.DataFrame]:
        if (year, month) in self.cached_csv_data:
            return self.cached_csv_data[(year, month)]
            
        filename = f"{year}_{month:02d}.csv"
        path = os.path.join(self.csv_dir, filename)
        
        if not os.path.exists(path):
            return None
            
        try:
            # Parse
            # JULIAN DAY UTC,YEAR,Month ,Day,,... Time UTC, ..., Elevation DEG, ..., Azimuth DEG
            df = pd.read_csv(path)
            
            # Construct Datetime Index
            # Format: 'Time UTC' is 'HH:MM'. 
            # 'YEAR', 'Month ', 'Day'. 
            # Warning: 'Month ' has space?
            
            col_map = {c: c.strip() for c in df.columns}
            df.rename(columns=col_map, inplace=True)
            
            # Create datetime string
            # "2025-12-01 00:00"
            # df['FullTime'] = df['YEAR'].astype(str) + '-' + df['Month'].astype(str) + '-' + df['Day'].astype(str) + ' ' + df['Time UTC']
            
            # Optimized parsing
            df['datetime'] = pd.to_datetime(
                df[['YEAR', 'Month', 'Day']].astype(str).agg('-'.join, axis=1) + ' ' + df['Time UTC']
            )
            
            # Set Index
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            # Keep only Az/El
            clean_df = df[['Azimuth DEG', 'Elevation DEG']].astype(float)
            
            # Cache
            self.cached_csv_data[(year, month)] = clean_df
            return clean_df
            
        except Exception as e:
            print(f"Error parsing sun CSV {path}: {e}")
            return None
