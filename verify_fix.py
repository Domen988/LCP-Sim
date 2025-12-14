
import numpy as np
from lcp.core.geometry import PanelGeometry
from lcp.core.config import ScenarioConfig
from lcp.simulation import SimulationRunner
from lcp.physics.engine import InfiniteKernel

def verify():
    print("--- VERIFY INFINITE NEIGHBORS ---")
    
    # 1. Setup
    geo = PanelGeometry(width=1.0, length=1.0, pivot_offset=(0,0,0.05))
    cfg = ScenarioConfig(grid_pitch_x=1.5, grid_pitch_y=1.5, total_panels=9) 
    
    runner = SimulationRunner("dummy.csv")
    runner.kernel.geo = geo
    runner.kernel.cfg = cfg
    
    # 2. Kernel Setup
    runner.kernel.pivots = {}
    for r in range(3):
        for c in range(3):
            y = (r - 1) * cfg.grid_pitch_y
            x = (c - 1) * cfg.grid_pitch_x
            runner.kernel.pivots[(r,c)] = np.array([float(x), float(y), 0.0])
            
    # 3. Sun Position
    # Afternoon / Summer (South Africa -> Sun North).
    # Sun Azimuth ~300 (NW)? Or 260?
    # User said "Afternoon". Shadows cast East/South.
    # So Sun is West/North.
    local_az = 300.0 
    sun_el = 45.0
    
    target_rot = runner.kernel.rig.get_orientation(local_az, sun_el)
    
    # 4. Check (1,1) Neighbors
    print("\n--- CHECKING NEIGHBORS (1,1) ---")
    pos_11 = runner.kernel.pivots[(1,1)]
    rot_11 = target_rot
    
    vn = runner.kernel._get_virtual_neighbors(1, 1, pos_11, rot_11)
    
    print(f"Total Virtual Neighbors: {len(vn)}")
    
    # Check for North Neighbor (dr=1, dc=0) -> Offset (0, 1.5)
    north_found = False
    for p, r in vn:
        off = p - pos_11
        if abs(off[0]) < 0.1 and abs(off[1] - 1.5) < 0.1:
            print("Found North Neighbor (0, 1.5)")
            north_found = True
            
    if not north_found:
        print("ERROR: North Neighbor MISSING!")
    else:
        print("North Neighbor Present.")

    # 5. Calculate Shadows
    # Convert Az/El to Vec (copy logic from engine if needed, or deduce)
    rad_az = np.radians(local_az)
    rad_el = np.radians(sun_el)
    # Note: Engine uses:
    # sun_vec = [sin(az)cos(el), cos(az)cos(el), sin(el)]
    # Az 0 = North (+Y). Az 90 = East (+X).
    # Az 300 = -60. sin(-60) = -0.86 (West). cos(-60) = 0.5 (North).
    # Matches "West/North".
    
    sun_vec = np.array([
        np.sin(rad_az) * np.cos(rad_el),
        np.cos(rad_az) * np.cos(rad_el),
        np.sin(rad_el)
    ])
    print(f"Sun Vec: {sun_vec}")
    
    loss, polys = runner.kernel.shadower.calculate_loss(pos_11, rot_11, vn, sun_vec)
    print(f"Shadow Loss: {loss}")
    print(f"Shadow Polys Count: {len(polys)}")
    for i, poly in enumerate(polys):
        print(f"Poly {i} vertices: {len(poly)}")

if __name__ == "__main__":
    verify()
