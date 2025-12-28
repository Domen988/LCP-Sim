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
        Legacy Scalar Method.
        """
        # Wrap the scalar input into arrays and use solve_timeseries
        # For full correctness including overrides, we might keep the old logic or adapt.
        # But for optimization, let's keep old logic intact to avoid breaking overrides that might change per step?
        # Actually overrides are usually fixed for a run scenario.
        # Let's keep the existing implementation of solve_timestep for safety/regression testing
        # and add solve_timeseries as a new high-speed path.
        return self._solve_timestep_scalar(sun_az, sun_el, enable_safety, inactive_override, active_override)

    def _solve_timestep_scalar(self, sun_az, sun_el, enable_safety, inactive_override, active_override):
        # ... (Previous Implementation Logic - pasted back to ensure continuity)
        # 1. Calculate Ideal Tracking Orientation
        min_el = getattr(self.cfg, 'min_elevation', 15.0)
        target_el = max(min_el, sun_el)
        target_rot = self.rig.get_orientation(sun_az, target_el)
            
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
                    rot_map[idx] = target_rot 
            else:
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
                    if manual_active_rot is not None:
                         mode = "MANUAL_ACTIVE"
                         rot = manual_active_rot
                
                panel_kinematics[idx] = (self.pivots[idx], rot, mode)
        
        # 4. Calculate Shadow Loss & Final State (Pass 2)
        states = []
        
        for r in range(3):
            for c in range(3):
                idx = (r,c)
                pos, rot, mode = panel_kinematics[idx]
                v_neighbors = self._get_virtual_neighbors(r, c, pos, rot)
                
                if collision_detected:
                    shad_loss = 0.0
                    shad_polys = []
                else:
                    shad_loss, shad_polys = self.shadower.calculate_loss(pos, rot, v_neighbors, sun_vec)
                
                normal = rot[:, 2]
                cos_theta = max(0.0, np.dot(normal, sun_vec))
                stow_loss = 1.0 if "STOW" in mode else 0.0
                
                if "STOW" in mode:
                    p_factor = 0.0 
                else:
                    pointing_threshold = 0.9999
                    if cos_theta < pointing_threshold:
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

    def solve_timeseries(self, az_array: np.ndarray, el_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Vectorized solver for the full time series.
        Returns dictionary of arrays: 'power_factor', 'collision', 'mode', etc.
        (Simplified: Assuming no manual overrides for now)
        """
        N = len(az_array)
        min_el = getattr(self.cfg, 'min_elevation', 15.0)
        
        # 1. Kinematics (Vectorized)
        target_el = np.maximum(min_el, el_array)
        
        # (N, 3, 3)
        target_rots = self.rig.get_orientation(az_array, target_el)
        stow_rots = self.rig.get_orientation(az_array, np.full(N, 45.0))
        
        # Sun Vectors (N, 3)
        phi_s = np.radians(az_array)
        theta_s = np.radians(90 - el_array)
        sx = np.sin(theta_s) * np.sin(phi_s)
        sy = np.sin(theta_s) * np.cos(phi_s)
        sz = np.cos(theta_s)
        sun_vecs = np.stack((sx, sy, sz), axis=-1)
        
        # 2. Collision Detection (Vectorized)
        # We assume all panels track target_rot.
        # Check adjacent pairs in 3x3 grid.
        # Pairs: Horizontal (0,0)-(0,1), etc. Vertical (0,0)-(1,0), etc.
        # Actually, since it's an infinite uniform grid assumption, we just need to check
        # one representative interaction types? 
        # But we have specific neighbors.
        # Let's check all unique neighbor pairs in 3x3.
        
        clash_mask = np.zeros(N, dtype=bool)
        
        idx_list = list(self.pivots.keys())
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                idx_i = idx_list[i]
                idx_j = idx_list[j]
                
                # Check Parallel Clash (Vectorized)
                # target_rots is shared by all panels in idealized tracking
                # Pass SAME rotation array for both A and B
                c_res = self.collider.check_clash(self.pivots[idx_i], self.pivots[idx_j], target_rots, rot_b=None)
                
                clash_mask |= c_res
                
        # 3. Hybrid Shadow Loop
        # We calculate shadows only where:
        # a) Sun is up (el > 0)
        # b) No collision (Safety stow -> Shadow=0)
        
        valid_mask = (el_array > 0) & (~clash_mask)
        valid_indices = np.where(valid_mask)[0]
        
        shadow_factors = np.zeros(N)
        
        # We pick the Center Panel (1,1) as the representative for yield
        center_idx = (1,1)
        center_pos = self.pivots[center_idx]
        
        # Pre-fetch neighbors offsets for (1,1) - usually all 8 neighbors
        neighbor_offsets = self.neighbor_map[center_idx]
        
        # Loop mainly for geometry
        for i in valid_indices:
            rot = target_rots[i] # (3,3)
            sun = sun_vecs[i]   # (3,)
            
            # Virtual neighbors for (1,1)
            v_neighbors = [(center_pos + off, rot) for off in neighbor_offsets]
            
            loss, _ = self.shadower.calculate_loss(center_pos, rot, v_neighbors, sun)
            shadow_factors[i] = loss
            
        # 4. Final Aggregation
        # Power Factor = CosTheta * (1 - Shadow) * Availability
        # Normal vector is 3rd column of rot
        # For Tracking: target_rots
        # For Stow: stow_rots
        
        # Construct final arrays
        final_rots = target_rots.copy()
        final_rots[clash_mask] = stow_rots[clash_mask]
        
        # Normals (N, 3)
        normals = final_rots[:, 2, :] # Access (N, 3, 3) -> (N, 3)
        
        # Dot Product (N, 3) . (N, 3) -> (N,)
        # einsum 'ij,ij->i'
        cos_theta = np.einsum('ij,ij->i', normals, sun_vecs)
        cos_theta = np.maximum(0.0, cos_theta)
        
        # Pointing Efficiency Threshold
        pointer_mask = cos_theta < 0.9999
        
        power_factors = cos_theta * (1.0 - shadow_factors)
        
        # Apply masks
        # If Stow (clash): Power=0
        power_factors[clash_mask] = 0.0
        # If Bad Pointing: Power=0 (Only if not stowed? Stow is 0 anyway)
        power_factors[pointer_mask] = 0.0
        
        return {
            'power_factors': power_factors,
            'collision_mask': clash_mask,
            'shadow_factors': shadow_factors,
            'sun_vecs': sun_vecs
        }
