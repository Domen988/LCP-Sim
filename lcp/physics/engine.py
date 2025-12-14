import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from lcp.core.geometry import PanelGeometry
from lcp.core.config import ScenarioConfig
from lcp.physics.kinematics import AzElRig, TrackerRig
from lcp.physics.collisions import CollisionDetector
from lcp.physics.shadows import ShadowEngine

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
    shadow_polys: List[np.ndarray] # World-space polygon vertices
    power_factor: float # Final multiplier
    
    @property
    def normal(self) -> np.ndarray:
        return self.rotation[:, 2]

class InfiniteKernel:
    """
    3x3 Kernel + Infinite Virtual Neighbors for accurate shadowing.
    """
    def __init__(self, geo: PanelGeometry, cfg: ScenarioConfig):
        self.geo = geo
        self.cfg = cfg
        self.rig: TrackerRig = AzElRig()
        self.collider = CollisionDetector(geo, cfg)
        self.shadower = ShadowEngine(geo)
        
        # Initialize pivot positions
        self.pivots: Dict[Tuple[int,int], np.ndarray] = {}
        for r in range(3):
            for c in range(3):
                y = (r - 1) * cfg.grid_pitch_y
                x = (c - 1) * cfg.grid_pitch_x
                self.pivots[(r,c)] = np.array([float(x), float(y), 0.0])

    def _get_virtual_neighbors(self, r, c, current_pos, current_rot):
        neighbors = []
        # Expand kernel to capture long shadows
        # Radius 1 (Immediate Neighbors) for Performance (Speedup ~3x over Radius 2)
        
        radius = 1
        
        # r=0 (South Edge) -> dr >= 0
        # r=1 (Interior) -> dr in -radius..radius
        # r=2 (North Edge) -> dr <= 0
        dr_min = 0 if r == 0 else -radius
        dr_max = 0 if r == 2 else radius
        
        # c=0 (Left Edge) -> dc >= 0
        # c=1 (Center) -> dc in -radius..radius
        # c=2 (Right Edge) -> dc <= 0
        dc_min = 0 if c == 0 else -radius
        dc_max = 0 if c == 2 else radius
        
        px = self.cfg.grid_pitch_x
        py = self.cfg.grid_pitch_y
        
        for dr in range(dr_min, dr_max + 1):
            for dc in range(dc_min, dc_max + 1):
                if dr == 0 and dc == 0: continue
                
                # Coordinate: Neighbor relative to Current
                # dx = dc * px
                # dy = dr * py
                offset = np.array([float(dc * px), float(dr * py), 0.0])
                n_pos = current_pos + offset
                
                neighbors.append((n_pos, current_rot))
        
        return neighbors

    def solve_timestep(self, sun_az: float, sun_el: float, enable_safety: bool = True) -> Tuple[List[PanelState], bool]:
        """
        Solves the state of the 3x3 kernel for a given sun position.
        """
        # 1. Calculate Ideal Tracking Orientation
        if sun_el <= 0:
            target_rot = self.rig.get_orientation(sun_az, max(0.01, sun_el))
        else:
            target_rot = self.rig.get_orientation(sun_az, sun_el)
            
        # 2. Check Collisions
        collision_detected = False
        idx_list = list(self.pivots.keys())
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                if self.collider.check_clash(self.pivots[idx_list[i]], self.pivots[idx_list[j]], target_rot):
                    collision_detected = True
                    break
            if collision_detected:
                break
                
        # 3. Determine Modes & Kinematics (Pass 1)
        stow_rot = self.rig.get_orientation(sun_az, 45.0)
        
        rad_az = np.radians(sun_az)
        rad_el = np.radians(sun_el)
        sun_vec = np.array([
            np.sin(rad_az) * np.cos(rad_el),
            np.cos(rad_az) * np.cos(rad_el),
            np.sin(rad_el)
        ])
        
        panel_kinematics = {}
        
        for r in range(3):
            for c in range(3):
                idx = (r,c)
                is_even = ((r + c) % 2 == 0)
                
                if collision_detected and is_even and enable_safety:
                    mode = "STOW"
                    rot = stow_rot
                else:
                    mode = "TRACKING"
                    rot = target_rot
                
                panel_kinematics[idx] = (self.pivots[idx], rot, mode)
        
        # 4. Calculate Shadow Loss & Final State (Pass 2)
        states = []
        
        for r in range(3):
            for c in range(3):
                idx = (r,c)
                pos, rot, mode = panel_kinematics[idx]
                
                # Use Virtual Neighbors for infinite plant emulation
                # Note: 'rot' is uniform (Safety=False) or We assume neighbor rotates same (good approx)
                v_neighbors = self._get_virtual_neighbors(r, c, pos, rot)
                
                # Shadow Loss
                if collision_detected:
                    shad_loss = 0.0
                    shad_polys = []
                else:
                    # Pass virtual neighbors instead of finite kernel neighbors
                    shad_loss, shad_polys = self.shadower.calculate_loss(pos, rot, v_neighbors, sun_vec)
                
                # Cosine Loss
                normal = rot[:, 2]
                cos_theta = max(0.0, np.dot(normal, sun_vec))
                
                stow_loss = 1.0 if mode == "STOW" else 0.0
                
                if mode == "STOW":
                    # User Rule: STOWED panels generate 0.
                    p_factor = 0.0 
                else:
                    p_factor = cos_theta * (1.0 - shad_loss)
                    
                states.append(PanelState(
                    index=idx,
                    position=pos,
                    rotation=rot,
                    mode=mode,
                    collision=collision_detected,
                    theta_loss=1.0 - cos_theta,
                    stow_loss=stow_loss,
                    shadow_loss=shad_loss,
                    shadow_polys=shad_polys,
                    power_factor=p_factor
                ))
                
        return states, collision_detected
