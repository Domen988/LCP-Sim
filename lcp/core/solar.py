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
        
    def get_position(self, dt: datetime | pd.DatetimeIndex) -> SunPosition | pd.DataFrame:
        """
        Calculates high-precision solar position.
        Accepts scalar datetime or DatetimeIndex.
        Returns SunPosition (scalar) or DataFrame (columns: azimuth, elevation).
        """
        is_scalar = isinstance(dt, (datetime, pd.Timestamp))
        
        ts = dt
        if is_scalar:
            ts = pd.Timestamp(dt)
            if ts.tz is None:
                ts = ts.tz_localize(self.loc.tz)
            else:
                ts = ts.tz_convert(self.loc.tz)
        else:
            # Vectorized Case
            if ts.tz is None:
                ts = ts.tz_localize(self.loc.tz)
            else:
                ts = ts.tz_convert(self.loc.tz)
            
        pos = self.loc.get_solarposition(ts)
        
        if is_scalar:
            return SunPosition(
                azimuth=float(pos['azimuth'].iloc[0]), 
                elevation=float(pos['elevation'].iloc[0])
            )
        else:
            return pos[['azimuth', 'elevation']]
