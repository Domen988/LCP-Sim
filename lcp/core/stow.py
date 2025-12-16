
import json
import bisect
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from datetime import datetime

@dataclass
class Keyframe:
    timestamp: datetime
    az: float
    el: float

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "az": self.az,
            "el": self.el
        }

    @staticmethod
    def from_dict(data):
        return Keyframe(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            az=data["az"],
            el=data["el"]
        )

class StowProfile:
    def __init__(self, profile_name: str = "New_Profile", limit_over_the_top: bool = False):
        self.profile_name = profile_name
        self.limit_over_the_top = limit_over_the_top
        self.keyframes: List[Keyframe] = []

    def sort_keyframes(self):
        """Ensures keyframes are sorted by timestamp."""
        self.keyframes.sort(key=lambda k: k.timestamp)

    def add_keyframe(self, timestamp: datetime, az: float, el: float):
        """Adds or updates a keyframe at the given timestamp."""
        # Check if keyframe exists at exact time (rare but possible)
        for k in self.keyframes:
            if k.timestamp == timestamp:
                k.az = az
                k.el = el
                return
        
        # New keyframe
        self.keyframes.append(Keyframe(timestamp, az, el))
        self.sort_keyframes()

    def remove_keyframe(self, index: int):
        """Removes keyframe by index."""
        if 0 <= index < len(self.keyframes):
            self.keyframes.pop(index)

    def get_position_at(self, query_time: datetime) -> Optional[Tuple[float, float]]:
        """
        Calculates interpolated (az, el) at query_time.
        Returns None if no keyframes exist.
        Returns nearest keyframe if outside range.
        Perfoms Linear Interpolation between two nearest keyframes.
        """
        if not self.keyframes:
            return None

        # 1. Check Bounds
        if query_time <= self.keyframes[0].timestamp:
            k = self.keyframes[0]
            return (k.az, k.el)
        if query_time >= self.keyframes[-1].timestamp:
            k = self.keyframes[-1]
            return (k.az, k.el)

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
            return (k_prev.az, k_prev.el)

        elapsed_sec = (query_time - k_prev.timestamp).total_seconds()
        alpha = elapsed_sec / total_sec

        # Linear Interpolation
        # Note: Azimuth wrapping arithmetic (e.g. 350 -> 10) is NOT handled here.
        # User is expected to provide unwrapped values if needed (e.g. -10 to 10), 
        # or we accept linear transit through non-optimal path.
        # Given the "Manual" nature, we assume WYSIWYG linear mixing.
        
        az = k_prev.az + alpha * (k_next.az - k_prev.az)
        el = k_prev.el + alpha * (k_next.el - k_prev.el)

        return (az, el)

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
