
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from lcp.core.geometry import PanelGeometry
from lcp.core.config import ScenarioConfig
from lcp.physics.engine import InfiniteKernel
from lcp.physics.shadows import ShadowEngine
from lcp.simulation import SimulationRunner

def debug_shadows():
    # 1. Setup
    print("--- DEBUG SHADOW SLANT ---")
    
    geo = PanelGeometry(width=1.0, length=1.0, pivot_offset=(0,0,0.05))
    cfg = ScenarioConfig(grid_pitch_x=1.01, grid_pitch_y=1.01, total_panels=9) # 1cm Gap
    
    runner = SimulationRunner("dummy.csv") # Init runner
    runner.kernel.geo = geo
    runner.kernel.cfg = cfg
    runner.kernel.collider.geo = geo
    runner.kernel.collider.cfg = cfg
    
    # 2. Kernel setup
    runner.kernel.pivots = {}
    for r in range(3):
        for c in range(3):
            y = (r - 1) * cfg.grid_pitch_y
            x = (c - 1) * cfg.grid_pitch_x
            runner.kernel.pivots[(r,c)] = np.array([float(x), float(y), 0.0])
            
    # 3. Force Afternoon Sun
    local_az = 260.0 
    sun_el = 10.0 # Very Low sun
    
    # Calculate Sun Vector
    # North=Y, East=X, Up=Z
    # Azimuth 0 = North. 90 = East. 270 = West.
    # Convert Az to vector.
    az_rad = np.radians(local_az)
    el_rad = np.radians(sun_el)
    
    # Note: LCP coordinates might differ? 
    # Usually: X=East, Y=North.
    # If Az=0 is North (Y).
    # Then X = sin(Az), Y = cos(Az).
    sys_x = np.sin(az_rad) * np.cos(el_rad)
    sys_y = np.cos(az_rad) * np.cos(el_rad)
    sys_z = np.sin(el_rad)
    sun_vec = np.array([sys_x, sys_y, sys_z])
    
    print(f"Sun Vector (Az={local_az}, El={sun_el}): {sun_vec}")
    
    # 4. Solve State
    # Safety=False (Tracking)
    states, collision = runner.kernel.solve_timestep(local_az, sun_el, enable_safety=False)
    
    # 5. Inspect Shadows
    # (0,1) is North Middle. Coordinate (0, 0). (Wait, (0,1) is Top-Middle. r=0 -> y = -1.05)
    # (1,1) is Center. Coordinate (0, 0).
    # (2,1) is South Middle. Coordinate (0, 1.05).
    
    pivot_01 = runner.kernel.pivots[(0,1)]
    pivot_11 = runner.kernel.pivots[(1,1)]
    pivot_21 = runner.kernel.pivots[(2,1)]
    print(f"Pos (0,1) [NorthMid]: {pivot_01}")
    print(f"Pos (1,1) [Center]: {pivot_11}")
    print(f"Pos (2,1) [SouthMid]: {pivot_21}")
    
    k_map = {s.index: s for s in states}
    
    def print_shadow(idx):
        s = k_map.get(idx)
        if not s: return
        print(f"\n--- Panel {idx} ---")
        print(f"Rotation Matrix:\n{s.rotation}")
        if not s.shadow_polys:
            print("No Shadows.")
            return
            
        for i, poly in enumerate(s.shadow_polys):
            print(f"Shadow Poly {i}:")
            # Check edge vectors
            # Expected: Top and Bottom edges should be roughly parallel to Panel Y-axis (which is rotated Y-axis)
            # Or for simple tracking, axis is N-S (Y). Panel rotates around Y.
            # Side edges (East/West) are parallel to Y-axis.
            # Top/Bot edges are parallel to X-axis (Width).
            # Wait. Panel dimensions: Width (X), Length (Y).
            # Tracker Axis is Y.
            # So "Side" edges are parallel to Y. Top/Bot edges are parallel to X (roughly).
            
            # Shadow from West neighbor (Parallel).
            # Casting edge is East edge of neighbor (Parallel to Y).
            # Shadow top edge on receiver should be parallel to Y.
            
            for v in poly:
                print(f"  {v}")

    print_shadow((0,1))
    print_shadow((1,1))
    print_shadow((2,1))

if __name__ == "__main__":
    debug_shadows()
