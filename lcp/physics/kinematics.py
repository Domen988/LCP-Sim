from typing import Protocol, Tuple
import numpy as np

class TrackerRig(Protocol):
    def get_orientation(self, sun_az: float, sun_el: float) -> np.ndarray:
        """
        Returns the 3x3 rotation matrix for the panel given sun position.
        """
        ...

class AzElRig:
    """
    Standard Azimuth-Elevation tracking rig.
    Aligns the panel normal vector directly to the sun vector.
    """
    def get_orientation(self, sun_az_deg: float | np.ndarray, sun_el_deg: float | np.ndarray) -> np.ndarray:
        # Convert to radians
        # Ensure array inputs for vectorization
        az = np.radians(np.atleast_1d(sun_az_deg))
        el = np.radians(np.atleast_1d(sun_el_deg))
        
        # Check if scalar input to unpack output later
        is_scalar = np.ndim(sun_az_deg) == 0 and np.ndim(sun_el_deg) == 0

        phi_s = az # Azimuth (CW from North)
        theta_s = np.radians(90) - el # Zenith angle
        
        # Spherical to Cartesian (X=East, Y=North, Z=Up)
        x = np.sin(theta_s) * np.sin(phi_s)
        y = np.sin(theta_s) * np.cos(phi_s)
        z = np.cos(theta_s)
        
        # Vectorized stacking: (N, 3)
        # Transpose so rows are vectors for now? No, let's keep shape (N, 3) or (3, N)
        # stack arrays: x is (N,), y is (N,), z is (N,)
        normal = np.stack((x, y, z), axis=-1) # (N, 3)

        # Target 'Right' vector (Local X).
        # Bearing = (sin(az), cos(az), 0).
        # Right = (cos(az), -sin(az), 0).
        
        right_x = np.cos(phi_s)
        right_y = -np.sin(phi_s)
        right_z = np.zeros_like(phi_s)
        
        right = np.stack((right_x, right_y, right_z), axis=-1) # (N, 3)
        
        # Zenith Check (Vectorized)
        # If z > 0.9999, forces right vector to East
        zenith_mask = np.abs(z) > 0.9999
        if np.any(zenith_mask):
            # Broadcast replacement
            right[zenith_mask] = np.array([1.0, 0.0, 0.0])

        # Target 'Up' vector (Local Y)
        # Y = Normal x Right? No: Y = Z x X (Normal x Right)
        # But earlier logic was: Y = Z x X.
        # Let's check: Normal (Z) x Right (X) = Y.
        # Yes, Up = Cross(Normal, Right)
        
        # Vectorized Cross Product
        up = np.cross(normal, right) # (N, 3)
        
        # The Rotation Matrix R = [X_col, Y_col, Z_col]
        # (N, 3, 3) stack
        R = np.stack((right, up, normal), axis=-1)
        
        if is_scalar:
            return R[0]
            
        return R

