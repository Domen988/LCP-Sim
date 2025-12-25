
import json
import bisect
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from datetime import datetime

@dataclass
class Keyframe:
    timestamp: datetime
    # Inactive Group (Stow)
    inactive_az: float
    inactive_el: float
    # Active Group (Tracking)
    active_az: float
    active_el: float
    # Reference Sun Position (for Reset)
    sun_az: float
    sun_el: float

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "inactive_az": self.inactive_az,
            "inactive_el": self.inactive_el,
            "active_az": self.active_az,
            "active_el": self.active_el,
            "sun_az": self.sun_az,
            "sun_el": self.sun_el
        }

    @staticmethod
    def from_dict(data):
        # Backward Compatibility
        if "az" in data:
            # Old format: az/el -> inactive_az/inactive_el
            # active -> 0 (or strictly speaking we should maybe try to estimate, 
            # but 0 ensures it's explicit "unknown/neutral"). 
            # Sun pos -> 0
            return Keyframe(
                timestamp=datetime.fromisoformat(data["timestamp"]),
                inactive_az=data["az"],
                inactive_el=data["el"],
                active_az=0.0,
                active_el=0.0,
                sun_az=0.0,
                sun_el=0.0
            )
            
        return Keyframe(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            inactive_az=data.get("inactive_az", 0.0),
            inactive_el=data.get("inactive_el", 0.0),
            active_az=data.get("active_az", 0.0),
            active_el=data.get("active_el", 0.0),
            sun_az=data.get("sun_az", 0.0),
            sun_el=data.get("sun_el", 0.0)
        )

class StowProfile:
    def __init__(self, profile_name: str = "New_Profile", limit_over_the_top: bool = False):
        self.profile_name = profile_name
        self.limit_over_the_top = limit_over_the_top
        self.keyframes: List[Keyframe] = []

    def sort_keyframes(self):
        """Ensures keyframes are sorted by timestamp."""
        self.keyframes.sort(key=lambda k: k.timestamp)

    def add_keyframe(self, timestamp: datetime, 
                     inactive_az: float, inactive_el: float,
                     active_az: float, active_el: float,
                     sun_az: float, sun_el: float):
        """Adds or updates a keyframe at the given timestamp."""
        # Check if keyframe exists at exact time (rare but possible)
        for k in self.keyframes:
            if k.timestamp == timestamp:
                k.inactive_az = inactive_az
                k.inactive_el = inactive_el
                k.active_az = active_az
                k.active_el = active_el
                k.sun_az = sun_az
                k.sun_el = sun_el
                return
        
        # New keyframe
        self.keyframes.append(Keyframe(timestamp, inactive_az, inactive_el, active_az, active_el, sun_az, sun_el))
        self.sort_keyframes()

    def remove_keyframe(self, index: int):
        """Removes keyframe by index."""
        if 0 <= index < len(self.keyframes):
            self.keyframes.pop(index)

    def get_position_at(self, query_time: datetime) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculates interpolated (inactive_az, inactive_el, active_az, active_el) at query_time.
        Returns None if no keyframes exist.
        Returns nearest keyframe if outside range.
        Perfoms Linear Interpolation between two nearest keyframes.
        
        Note: We do NOT interpolate Sun Position as it's just a reference for that specific keyframe.
        The caller usually knows the current sun position if they need it.
        """
        if not self.keyframes:
            return None

        # 1. Check Bounds
        if query_time <= self.keyframes[0].timestamp:
            k = self.keyframes[0]
            return (k.inactive_az, k.inactive_el, k.active_az, k.active_el)
        if query_time >= self.keyframes[-1].timestamp:
            k = self.keyframes[-1]
            return (k.inactive_az, k.inactive_el, k.active_az, k.active_el)

        # 2. Find Neighbors (bisect)
        # We want the index where query_time would be inserted
        times = [k.timestamp for k in self.keyframes]
        idx = bisect.bisect_right(times, query_time)
        
        # idx is the first element > query_time
        # so idx-1 is the element <= query_time
        k_prev = self.keyframes[idx - 1]
        k_next = self.keyframes[idx]

        # 3. Interpolate
        # Total duration between frames
        total_sec = (k_next.timestamp - k_prev.timestamp).total_seconds()
        if total_sec == 0:
            return (k_prev.inactive_az, k_prev.inactive_el, k_prev.active_az, k_prev.active_el)

        elapsed_sec = (query_time - k_prev.timestamp).total_seconds()
        alpha = elapsed_sec / total_sec

        # Linear Interpolation
        # Note: Azimuth wrapping arithmetic (e.g. 350 -> 10) is NOT handled here.
        # User is expected to provide unwrapped values if needed (e.g. -10 to 10), 
        # or we accept linear transit through non-optimal path.
        # Given the "Manual" nature, we assume WYSIWYG linear mixing.
        
        i_az = k_prev.inactive_az + alpha * (k_next.inactive_az - k_prev.inactive_az)
        i_el = k_prev.inactive_el + alpha * (k_next.inactive_el - k_prev.inactive_el)
        
        a_az = k_prev.active_az + alpha * (k_next.active_az - k_prev.active_az)
        a_el = k_prev.active_el + alpha * (k_next.active_el - k_prev.active_el)

        return (i_az, i_el, a_az, a_el)

    def save(self, filepath: str):
        data = {
            "profile_name": self.profile_name,
            "limit_over_the_top": self.limit_over_the_top,
            "keyframes": [k.to_dict() for k in self.keyframes]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load(filepath: str) -> 'StowProfile':
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        profile = StowProfile(
            profile_name=data.get("profile_name", "Unknown"),
            limit_over_the_top=data.get("limit_over_the_top", False)
        )
        
        raw_frames = data.get("keyframes", [])
        for rf in raw_frames:
            profile.keyframes.append(Keyframe.from_dict(rf))
        
        profile.sort_keyframes()
        return profile
