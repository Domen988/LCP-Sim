
import unittest
import os
import shutil
from datetime import datetime, timedelta
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from lcp.core.stow import StowProfile

class TestStowProfile(unittest.TestCase):
    def setUp(self):
        self.test_file = "lcp/core/dummy_profile.json"
        self.profile = StowProfile("TestProfile")
        # Base Time: 12:00
        self.t0 = datetime(2025, 1, 1, 12, 0, 0)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_interpolation_linear(self):
        """Test simple linear interpolation between two points."""
        print("Testing Linear Interpolation...")
        # 12:00 -> Az=0, El=0
        self.profile.add_keyframe(self.t0, 0.0, 0.0)
        # 13:00 -> Az=100, El=50
        self.profile.add_keyframe(self.t0 + timedelta(hours=1), 100.0, 50.0)

        # Query 12:30 (Midpoint)
        mid_time = self.t0 + timedelta(minutes=30)
        az, el = self.profile.get_position_at(mid_time)
        
        self.assertAlmostEqual(az, 50.0)
        self.assertAlmostEqual(el, 25.0)
        print("  -> OK")

    def test_persistence(self):
        """Test save and load."""
        print("Testing Persistence...")
        self.profile.add_keyframe(self.t0, 10.0, 20.0)
        self.profile.save(self.test_file)
        
        loaded = StowProfile.load(self.test_file)
        self.assertEqual(loaded.profile_name, "TestProfile")
        self.assertEqual(len(loaded.keyframes), 1)
        self.assertAlmostEqual(loaded.keyframes[0].az, 10.0)
        print("  -> OK")

    def test_bounds(self):
        """Test querying outside keyframe range."""
        print("Testing Bounds...")
        self.profile.add_keyframe(self.t0, 10.0, 10.0)
        self.profile.add_keyframe(self.t0 + timedelta(hours=1), 20.0, 20.0)

        # Before start
        az, el = self.profile.get_position_at(self.t0 - timedelta(hours=1))
        self.assertEqual(az, 10.0)

        # After end
        az, el = self.profile.get_position_at(self.t0 + timedelta(hours=2))
        self.assertEqual(az, 20.0)
        print("  -> OK")

if __name__ == '__main__':
    unittest.main()
