import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from pvlib.location import Location

@dataclass
class SunPosition:
    azimuth: float # Degrees, North=0, East=90
    elevation: float # Degrees, Horizon=0, Zenith=90

class SolarCalculator:
    """
    Calculates Sun Position for Koster, South Africa using PVLib.
    Lat: -25.8633
    Lon: 26.8983
    Timezone: Africa/Johannesburg (UTC+2)
    """
    def __init__(self, lat=-25.8633, lon=26.8983):
        self.loc = Location(lat, lon, tz='Africa/Johannesburg', name='Koster')
        
    def get_position(self, dt: datetime) -> SunPosition:
        """
        Calculates high-precision solar position.
        Accepts naive datetime (assumed Local) or aware datetime.
        """
        ts = pd.Timestamp(dt)
        if ts.tz is None:
            ts = ts.tz_localize(self.loc.tz)
            
        pos = self.loc.get_solarposition(ts)
        
        return SunPosition(
            azimuth=float(pos['azimuth'].iloc[0]), 
            elevation=float(pos['elevation'].iloc[0])
        )
