import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class SunPosition:
    azimuth: float # Degrees, North=0, East=90
    elevation: float # Degrees, Horizon=0, Zenith=90

class SolarCalculator:
    """
    Calculates Sun Position for Koster, South Africa.
    Lat: -25.8633
    Lon: 26.8983
    """
    def __init__(self, lat=-25.8633, lon=26.8983):
        self.lat = lat
        self.lon = lon
        
    def get_position(self, dt: datetime) -> SunPosition:
        """
        Approximate Solar Position Algorithm (low precision but sufficient for Phase 1).
        """
        # Day of year
        dn = dt.timetuple().tm_yday
        
        # Local Solar Time
        # Equation of Time (minutes)
        b = 360 * (dn - 81) / 365.0
        b_rad = np.radians(b)
        eot = 9.87 * np.sin(2*b_rad) - 7.53 * np.cos(b_rad) - 1.5 * np.sin(b_rad)
        
        # Time Correction Factor
        # (Lon - LocalMeridian) * 4
        # Standard meridian for SA is UTC+2 = 30E? No, SAST is UTC+2. 30 deg East.
        meridian = 30.0
        tc = 4 * (self.lon - meridian) + eot
        
        lst_min = dt.hour * 60 + dt.minute + tc
        # Hour Angle (deg)
        # 12:00 LST = 0 deg. Morning < 0.
        ha = (lst_min / 4.0) - 180.0
        
        # Declination
        decl = 23.45 * np.sin(np.radians(360 * (284 + dn) / 365.0))
        
        lat_rad = np.radians(self.lat)
        dec_rad = np.radians(decl)
        ha_rad = np.radians(ha)
        
        # Elevation
        sin_el = np.sin(lat_rad)*np.sin(dec_rad) + np.cos(lat_rad)*np.cos(dec_rad)*np.cos(ha_rad)
        el = np.degrees(np.arcsin(sin_el))
        
        # Azimuth
        # cos(az) = (sin(dec) - sin(el)sin(lat)) / (cos(el)cos(lat))
        try:
            cos_az = (np.sin(dec_rad) - sin_el * np.sin(lat_rad)) / (np.cos(np.radians(el)) * np.cos(lat_rad))
            cos_az = np.clip(cos_az, -1.0, 1.0)
            az = np.degrees(np.arccos(cos_az))
            
            # Logic to resolve 0-360
            if ha > 0:
                az = 360.0 - az
        except:
            az = 0.0
            
        return SunPosition(azimuth=az, elevation=el)
