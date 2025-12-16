
import unittest
import numpy as np
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from lcp.core.geometry import PanelGeometry
from lcp.core.config import ScenarioConfig
from lcp.physics.engine import InfiniteKernel

class TestPhysicsOverride(unittest.TestCase):
    def setUp(self):
        # Setup similar to dashboard
        self.geo = PanelGeometry(width=2.0, length=2.0, thickness=0.05, pivot_offset=(0,0,0.2))
        self.cfg = ScenarioConfig(grid_pitch_x=2.5, grid_pitch_y=4.0, tolerance=0.0) # Zero tolerance for strict checking
        self.kernel = InfiniteKernel(self.geo, self.cfg)
        
        # Setup Pivots (Standard 3x3)
        self.kernel.pivots = {}
        for r in range(3):
            for c in range(3):
                y = (r - 1) * self.cfg.grid_pitch_y
                x = (c - 1) * self.cfg.grid_pitch_x
                self.kernel.pivots[(r,c)] = np.array([float(x), float(y), 0.0])

    def test_override_clash(self):
        print("Testing Override Collision...")
        
        # Let's assume Pitch X = 1.0. Width = 2.0. They MUST clash even if parallel.
        self.cfg.grid_pitch_x = 1.0
        self.cfg.grid_pitch_y = 10.0 # Far apart in Y
        # Re-init kernel to update references if needed (CollisionDetector holds cfg ref)
        self.kernel = InfiniteKernel(self.geo, self.cfg)

        # Re-init pivots
        for r in range(3):
            for c in range(3):
                y = (r - 1) * self.cfg.grid_pitch_y
                x = (c - 1) * self.cfg.grid_pitch_x
                self.kernel.pivots[(r,c)] = np.array([float(x), float(y), 0.0])
                
        # Now they overlap geometrically even at rest.
        # Standard solve should detect collision.
        states, collision = self.kernel.solve_timestep(sun_az=0, sun_el=90, enable_safety=True)
        self.assertTrue(collision, "Should clash due to tight pitch")
        print("  -> Baseline Clash OK")
        
        # Now try Override.
        # Even: Az=0, El=90 (Flat Horizontal)
        # Odd: Az=0, El=90 (Flat Horizontal)
        # They should still clash (overlap in X).
        states, collision = self.kernel.solve_timestep(sun_az=0, sun_el=90, enable_safety=True, stow_override=(0.0, 90.0))
        self.assertTrue(collision, "Should clash in Override too")
        
        # Now verification of Heterogeneous check.
        # Set pitch back to 2.5 (Safe).
        self.cfg.grid_pitch_x = 2.5
        self.kernel = InfiniteKernel(self.geo, self.cfg)

        for r in range(3):
            for c in range(3):
                y = (r - 1) * self.cfg.grid_pitch_y
                x = (c - 1) * self.cfg.grid_pitch_x
                self.kernel.pivots[(r,c)] = np.array([float(x), float(y), 0.0])
                
        # Sun Az=0, El=90 (Tracking Flat). Safe.
        states, collision = self.kernel.solve_timestep(sun_az=0, sun_el=90, enable_safety=True)
        self.assertFalse(collision, "Baseline Safe")
        
        # Let's ensure it returns correct result (False).
        states, collision = self.kernel.solve_timestep(sun_az=0, sun_el=90, enable_safety=True, stow_override=(0.0, 90.0))
        # Sun is 90 -> Tracking is Flat. Override 90 -> Manual is Flat.
        # Should be Safe.
        self.assertFalse(collision, "Override Safe")
        print("  -> Override Safe OK")

if __name__ == '__main__':
    unittest.main()
