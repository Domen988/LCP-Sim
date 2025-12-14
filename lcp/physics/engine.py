import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from lcp.core.geometry import PanelGeometry
from lcp.core.config import ScenarioConfig
from lcp.physics.kinematics import AzElRig, TrackerRig
from lcp.physics.collisions import CollisionDetector
from lcp.physics.shadows import ShadowCalculator, NoShadow

@dataclass
class PanelState:
    index: Tuple[int, int] # (row, col)
    position: np.ndarray
    rotation: np.ndarray # 3x3
    mode: str # "TRACKING", "STOW"
    collision: bool
    theta_loss: float # Cosine loss factor
    stow_loss: float # 1.0 if stow, 0.0 otherwise (fraction lost)
    shadow_loss: float
    power_factor: float # Final multiplier

    @property
    def normal(self) -> np.ndarray:
        # R * [0,0,1] is the 3rd column
        return self.rotation[:, 2]

class Kernel3x3:
    def __init__(self, geo: PanelGeometry, cfg: ScenarioConfig):
        self.geo = geo
        self.cfg = cfg
        self.rig: TrackerRig = AzElRig()
        self.collider = CollisionDetector(geo, cfg)
        self.shadower: ShadowCalculator = NoShadow()
        
        # Initialize pivot positions
        # Row 0..2 (Y), Col 0..2 (X)
        self.pivots: Dict[Tuple[int,int], np.ndarray] = {}
        for r in range(3):
            for c in range(3):
                # (1,1) is at 0,0,0
                y = (r - 1) * cfg.grid_pitch_y
                x = (c - 1) * cfg.grid_pitch_x
                self.pivots[(r,c)] = np.array([float(x), float(y), 0.0])

    def solve_timestep(self, sun_az: float, sun_el: float, enable_safety: bool = True) -> Tuple[List[PanelState], bool]:
        """
        Solves the state of the 3x3 kernel for a given sun position.
        Returns list of states and a boolean indicating if Safety Mode was triggered.
        """
        # 1. Calculate Ideal Tracking Orientation
        if sun_el <= 0:
            # Night / Horizon
            # Just default to Flat or Stow?
            # Spec doesn't say. Let's assume Flat (Zenith) for simplicity or Keep Tracking.
            # Usually stow at night. Let's assume Tracking (will be flat-ish or horizon).
            # Let's clip El to eps to avoid math errors if any
            target_rot = self.rig.get_orientation(sun_az, max(0.01, sun_el))
        else:
            target_rot = self.rig.get_orientation(sun_az, sun_el)
            
        # 2. Check Collisions (Assuming all Tracking)
        # We need to check neighbors.
        # Neighbors of (r,c) are (r+/-1, c+/-1)
        # Sufficient to iterate all panels and check their valid forward neighbors to avoid dupes?
        # Or just check center against neighbors? 
        # Spec: "If the Collision Detector flags *any* clash in the 3x3 kernel."
        # This implies checking the whole web.
        # Efficient way: Check (r,c) vs (r, c+1) and (r, c) vs (r+1, c).
        # We want to catch X-clashes and Y-clashes.
        # Also Diagonal clashes? "Panel Geometry" is box. Diagonal neighbors can clash if spun?
        # Check all unique pairs dist < limit.
        # 3x3 is small (9 items). 36 pairs. Brute force is fine (cheap).
        
        collision_detected = False
        
        # We only check collision if we are attempting to TRACK.
        # If we stow, we assume safe (as per Spec Phase 1).
        
        idx_list = list(self.pivots.keys())
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                idx_a = idx_list[i]
                idx_b = idx_list[j]
                
                # Check distances first as quick cull?
                # Detector is fast.
                if self.collider.check_clash(self.pivots[idx_a], self.pivots[idx_b], target_rot):
                    collision_detected = True
                    break
            if collision_detected:
                break
                
        # 3. Determine Modes
        # Checkerboard Stow: Even (0,0; 0,2...) Stow. Odd Track.
        # Even means (r+c) % 2 == 0?
        # (0,0) -> 0 even. (0,1) -> 1 odd. (1,1) -> 2 even.
        # Spec: "Even Panels (0,0; 0,2; 1,1...): Move to Stow Angle (45 deg Elevation)."
        
        states = []
        
        # Stow Rotation
        # Spec: Stow Angle = 45 deg Elevation.
        # Azimuth? Usually same as Sun or specific Stow Azimuth.
        # Let's assume Stow matches Sun Azimuth but fixed El=45.
        stow_rot = self.rig.get_orientation(sun_az, 45.0)
        
        sun_vec_flat = np.array([
            np.sin(np.radians(90-sun_el)) * np.sin(np.radians(sun_az)),
            np.sin(np.radians(90-sun_el)) * np.cos(np.radians(sun_az)),
            np.cos(np.radians(90-sun_el))
        ]) if sun_el > 0 else np.array([0,0,1]) # Approximate
        
        for r in range(3):
            for c in range(3):
                is_even = ((r + c) % 2 == 0)
                
                # Check collision AND safety enabled
                if collision_detected and is_even and enable_safety:
                    mode = "STOW"
                    rot = stow_rot
                    stow_loss = 1.0 # 0 power
                else:
                    mode = "TRACKING"
                    rot = target_rot
                    stow_loss = 0.0
                    
                # Calculate Power Factors
                # 1. Cosine Loss (Alignment)
                # Normal . Sun vector
                # If Tracking perfect, dot = 1.
                # If Stow, dot < 1.
                normal = rot[:, 2]
                cos_theta = np.dot(normal, sun_vec_flat)
                cos_theta = max(0.0, cos_theta) # Clamp negative
                
                # 2. Shadow Loss
                # Call Interface
                shad_loss = self.shadower.calculate_loss((r,c), [], sun_vec_flat, {})
                
                # Total Power Factor
                # If Stowed (Safety), Spec says "0 Watts". 
                # "Stowed panels generate 0 Watts."
                
                if mode == "STOW":
                    p_factor = 0.0
                else:
                    p_factor = cos_theta * (1.0 - shad_loss)
                    
                s = PanelState(
                    index=(r,c),
                    position=self.pivots[(r,c)],
                    rotation=rot,
                    mode=mode,
                    collision=collision_detected,
                    theta_loss=1.0 - cos_theta, # Just for storage?
                    stow_loss=stow_loss,
                    shadow_loss=shad_loss,
                    power_factor=p_factor
                )
                states.append(s)
                
        return states, collision_detected
