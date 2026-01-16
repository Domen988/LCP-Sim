
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
from lcp.core.persistence import PersistenceManager

class ContourMapper:
    """
    Generates a Safe Elevation Contour Map (LUT) from simulation data.
    Identifies the lowest elevation that causes a clash for each azimuth
    and defines the safe boundary slightly below it.
    """
    
    def __init__(self, output_dir: str = "Safe Elevation Contours"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def generate_map(self, sim_path: str, sim_name: str, bin_size: float = 1.0) -> str:
        """
        Scans simulation data for clashes and generates a header + Clash Contour JSON.
        Returns the path to the saved JSON file.
        
        Args:
            sim_path: Path to simulation directory
            sim_name: Name of simulation
            bin_size: Azimuth bin size in degrees for smoothing (default 1.0)
        """
        config_path = os.path.join(sim_path, "config.json")
        csv_path = os.path.join(sim_path, "timeseries.csv")
        
        if not os.path.exists(config_path) or not os.path.exists(csv_path):
            raise FileNotFoundError(f"Simulation files missing in {sim_path}")
            
        # 1. Load Header Info
        with open(config_path, "r") as f:
            cfg_data = json.load(f)
            
        header = {
            "source_simulation": sim_name,
            "generated_at": datetime.now().isoformat(),
            "geometry": cfg_data.get("geometry", {}),
            "config": cfg_data.get("config", {}),
            "bin_size": bin_size
        }
        
        # 2. Load Data
        df = pd.read_csv(csv_path)
        
        # Ensure correct types
        # 'safety' is True for CLASH, False for SAFE
        if df['safety'].dtype == 'O':
             is_clash = df['safety'].astype(str).str.lower() == 'true'
        else:
             is_clash = df['safety'].astype(bool)
             
        df['is_clash'] = is_clash
        
        # Filter only Clashes
        clash_df = df[df['is_clash']].copy()
        
        if clash_df.empty:
            raise ValueError("No clashes detected in this simulation. Cannot generate clash contour.")
            
        # 3. Clash Contour Logic (Smoothed)
        # Group clashes by Azimuth Bin.
        # Find Lowest Clash Elevation in that bin.
        # Store as Clash Contour (The "Danger Line").
        
        # Binning
        clash_df['az_bin'] = (clash_df['sun_az'] / bin_size).round() * bin_size
        
        # Group by Bin
        grouped = clash_df.groupby('az_bin')['sun_el'].min()
        
        contour = []
        for az, min_clash_el in grouped.items():
            # DIRECT CLASH VALUE - NO BUFFER
            contour.append([float(az), float(min_clash_el)])
            
        # Sort by Azimuth
        contour.sort(key=lambda x: x[0])
        
        output_data = {
            "header": header,
            "clash_contour": contour
        }
        
        output_filename = f"{sim_name}_contour.json"
        output_full_path = os.path.join(self.output_dir, output_filename)
        
        with open(output_full_path, "w") as f:
            json.dump(output_data, f, indent=2)
            
        return output_full_path
