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
        
        # Pre-compute neighbor offsets for optimization
        self.neighbor_offsets = []
        px = cfg.grid_pitch_x
        py = cfg.grid_pitch_y
        radius = 1
        
        # We pre-calculate offsets for all possible 9 positions (r,c)
        # Actually, the logic depends on (r,c) to know which neighbors exist (e.g. edge vs interior).
        # We can store a lookup table: (r,c) -> list of offsets (np.array)
        
        self.neighbor_map = {}
        
        for r in range(3):
            for c in range(3):
                # Logic from _get_virtual_neighbors
                dr_min = 0 if r == 0 else -radius
                dr_max = 0 if r == 2 else radius
                dc_min = 0 if c == 0 else -radius
                dc_max = 0 if c == 2 else radius
                
                offsets = []
                for dr in range(dr_min, dr_max + 1):
                    for dc in range(dc_min, dc_max + 1):
                        if dr == 0 and dc == 0: continue
                        off = np.array([float(dc * px), float(dr * py), 0.0])
                        offsets.append(off)
                self.neighbor_map[(r,c)] = offsets

        # Initialize pivot positions
        self.pivots: Dict[Tuple[int,int], np.ndarray] = {}
        for r in range(3):
            for c in range(3):
                y = (r - 1) * cfg.grid_pitch_y
                x = (c - 1) * cfg.grid_pitch_x
                self.pivots[(r,c)] = np.array([float(x), float(y), 0.0])

    def _get_virtual_neighbors(self, r, c, current_pos, current_rot):
        # Optimized lookup
        # returns list of (pos, rot)
        offsets = self.neighbor_map[(r,c)]
        # Broadcasting would be faster but loop is small (max 8)
        return [(current_pos + off, current_rot) for off in offsets]

    def solve_timestep(self, sun_az: float, sun_el: float, enable_safety: bool = True, 
                       inactive_override: Optional[Tuple[float, float]] = None,
                       active_override: Optional[Tuple[float, float]] = None) -> Tuple[List[PanelState], bool]:
        """
        Solves the state of the 3x3 kernel for a given sun position.
        inactive_override: (az, el) for Even panels (Stow Group).
        active_override: (az, el) for Odd panels (Active/Tracking Group).
        """
        # 1. Calculate Ideal Tracking Orientation
        if sun_el <= 0:
            target_rot = self.rig.get_orientation(sun_az, max(0.01, sun_el))
        else:
            target_rot = self.rig.get_orientation(sun_az, sun_el)
            
        # Prepare Manual Rotations
        manual_inactive_rot = None
        if inactive_override:
            manual_inactive_rot = self.rig.get_orientation(*inactive_override)
            
        manual_active_rot = None
        if active_override:
            manual_active_rot = self.rig.get_orientation(*active_override)

        # 2. Check Collisions
        collision_detected = False
        idx_list = list(self.pivots.keys())
        
        # Determine rotations for collision check
        rot_map = {}
        for idx in idx_list:
            r, c = idx
            is_stow_group = (r + c) % 2 == 0
            
            if is_stow_group:
                if manual_inactive_rot is not None:
                    rot_map[idx] = manual_inactive_rot
                else:
                    rot_map[idx] = target_rot # Default to Track if no override
            else:
                # Active Group
                if manual_active_rot is not None:
                    rot_map[idx] = manual_active_rot
                else:
                     rot_map[idx] = target_rot

        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                idx_i = idx_list[i]
                idx_j = idx_list[j]
                
                r_i = rot_map[idx_i]
                r_j = rot_map[idx_j]
                
                # Check Clash
                if np.array_equal(r_i, r_j): 
                    c_res = self.collider.check_clash(self.pivots[idx_i], self.pivots[idx_j], r_i, rot_b=None)
                else:
                    c_res = self.collider.check_clash(self.pivots[idx_i], self.pivots[idx_j], r_i, rot_b=r_j)
                
                if c_res:
                    collision_detected = True
                    break
            if collision_detected:
                break
                
        # 3. Determine Modes & Kinematics (Pass 1)
        # Default Auto-Stow
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
                is_stow_group = ((r + c) % 2 == 0)
                
                # Logic Priority:
                # 1. Manual Override (Active or Inactive)
                # 2. Safety Stow (if collision & safety on & is stow group)
                # 3. Tracking
                
                rot = target_rot
                mode = "TRACKING"
                
                if is_stow_group:
                    if manual_inactive_rot is not None:
                         mode = "MANUAL_INACTIVE"
                         rot = manual_inactive_rot
                    elif collision_detected and enable_safety:
                         mode = "STOW"
                         rot = stow_rot
                else:
                    # Active Group
                    if manual_active_rot is not None:
                         mode = "MANUAL_ACTIVE"
                         rot = manual_active_rot
                    # Even if collision, Active group tracks (unless we define Active Safety?)
                    # Current definitions say Active tracks sun always or manually overriden.
                
                panel_kinematics[idx] = (self.pivots[idx], rot, mode)
        
        # 4. Calculate Shadow Loss & Final State (Pass 2)
        states = []
        
        for r in range(3):
            for c in range(3):
                idx = (r,c)
                pos, rot, mode = panel_kinematics[idx]
                
                # Use Virtual Neighbors for infinite plant emulation
                v_neighbors = self._get_virtual_neighbors(r, c, pos, rot)
                
                if collision_detected:
                    shad_loss = 0.0
                    shad_polys = []
                else:
                    shad_loss, shad_polys = self.shadower.calculate_loss(pos, rot, v_neighbors, sun_vec)
                
                # Cosine Loss
                normal = rot[:, 2]
                cos_theta = max(0.0, np.dot(normal, sun_vec))
                
                stow_loss = 1.0 if "STOW" in mode else 0.0
                
                if "STOW" in mode:
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
