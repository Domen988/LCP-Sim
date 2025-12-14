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
    def get_orientation(self, sun_az_deg: float, sun_el_deg: float) -> np.ndarray:
        # Convert to radians
        az = np.radians(sun_az_deg)
        el = np.radians(sun_el_deg)
        
        # 1. Azimuth Rotation (Z-axis) - Standard physics convention (CCW from East) is often used,
        # but Solar Azimuth is usually CW from North (N=0, E=90).
        # We need to be careful with the coordinate system defined in the spec:
        # +X: East, +Y: North.
        # So Azimuth 0 (North) -> +Y axis. 
        # Azimuth 90 (East) -> +X axis.
        # We want to rotate the panel from Flat (Normal = +Z).
        
        # Let's define the rotations mathematically.
        # Panel default state: Flat, facing Up (+Z).
        # We need to point the normal vector towards the Sun.
        
        # In this specific rig (Az/El), the panel rotates around Z (Azimuth) and then X/Y (Elevation).
        # Actually for a typical tracker, the 'Azimuth' motor rotates the whole assembly around Vertical (Z).
        # The 'Elevation' motor rotates the panel around the horizontal axis.
        
        # Rotation around Z (Azimuth):
        # We want the projection of the normal to match the Sun's azimuth.
        # If Sun Az = 0 (North), Panel should point North.
        # Standard Rotation Matrix Rz(theta) rotates CCW.
        # If we stick to standard math: X=East, Y=North.
        # Az 0 (North) = 90 deg arithmetic.
        # This conversion is tricky.
        
        # Simpler approach: Construct the Target Vector and build rotation matrix from Basis Vectors.
        # Sun Vector S:
        # Sx = cos(el) * sin(az)  (East component)
        # Sy = cos(el) * cos(az)  (North component)
        # Sz = sin(el)            (Up component)
        
        # For an Az-El rig, the Panel Normal is exactly S.
        # We need a rotation matrix R such that R * [0,0,1]^T = S.
        # And we need to define the "Top" of the panel orientation (local Y).
        # Usually for Az-El, "up" on the panel (local Y) attempts to point to Zenith or align with the rig.
        
        # Let's derive R from Euler angles for Az-El rig.
        # 1. Rotate around Z by -Azimuth (if Az measures CW from North).
        # No, let's use the explicit Azimuth Motor angle.
        # If Sun is at Az, we rotate the rig to Az.
        # Then we tilt up by (90 - Elevation) from the horizon? Or just Elevation?
        # Usually Elevation is 0 at horizon, 90 at zenith. Panel is flat at El=90?
        # Actually, let's assume:
        # Panel starts flat (Normal=Z).
        # We want Normal to point at Sun.
        # Rotation 1: Yaw (Z) to match Azimuth.
        # Rotation 2: Pitch (Local X) to match Elevation.
        
        # Let's stick to valid matrices.
        # Sun at Az=0 (North), El=0 (Horizon). 
        # Panel Normal should be (0,1,0). 
        # Wait, if Panel Normal is (0,0,1) initially.
        # Rotate X by -90 deg -> Normal becomes (0,1,0).
        
        # Let's just implement the vector construction. It's more robust.
        # Z_local (Normal) = Sun Vector.
        # X_local (Right) = Cross(Z_global, Z_local).Normalized() -- Horizontal vector perpendicular to sun.
        # Y_local (Up) = Cross(Z_local, X_local).
        
        # Special case: Sun at Zenith (0,0,1).
        # Then Panel is flat. X_local = (1,0,0), Y_local = (0,1,0).
        
        phi_s = np.radians(sun_az_deg) # Azimuth (CW from North)
        theta_s = np.radians(90 - sun_el_deg) # Zenith angle
        
        # Spherical to Cartesian (X=East, Y=North, Z=Up)
        x = np.sin(theta_s) * np.sin(phi_s)
        y = np.sin(theta_s) * np.cos(phi_s)
        z = np.cos(theta_s)
        
        sun_vec = np.array([x, y, z])
        
        # Az-El Rig Constraint: The "Right" axis of the panel stays in the XY plane (horizontal).
        # This is the definition of Az-El tracking (elevation axis is horizontal).
        
        if abs(z) > 0.9999: # Zenith case
            return np.eye(3)
        
        z_prime = sun_vec # The new Normal
        
        # The axis of elevation rotation is horizontal and perpendicular to azimuth.
        # It corresponds to the local X axis?
        # Let's construct basis.
        # We want the panel to face the sun.
        # For Az-El rig, the sides of the panel stay horizontal? 
        # No, the bottom/top edge stays 'horizontal' relative to the elevation axis.
        # The elevation axis itself rotates with azimuth.
        
        # Let's define:
        # Global Z is Up.
        # Global Y is North.
        # Global X is East.
        
        # Rotation Z (Azimuth):
        # We rotate the whole frame so the Y-axis points to the Sun's bearing.
        # Rz = [[cosA, -sinA, 0], [sinA, cosA, 0], [0, 0, 1]]
        # But Az is CW from North. Standard rotation is CCW from X.
        # Angle from X (East) = 90 - Az.
        # Let's use simple logic:
        # We want to rotate vector (0,1,0) (North) to (sinAz, cosAz, 0).
        
        c = np.cos(phi_s)
        s = np.sin(phi_s)
        
        # Rotation matching North -> Sun Azimuth
        # R_az * [0,1,0] = [sin(az), cos(az), 0]
        # R_az = [ [c, s, 0], [-s, c, 0], [0, 0, 1] ]  <-- Check this?
        # [c s 0] * [0 1 0]T = s = sin(az) (X component). Correct.
        # [-s c 0] * [0 1 0]T = c = cos(az) (Y component). Correct.
        
        R_az = np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])
        # Wait, usually the row/col definition matters.
        # Let's stick to: Vector_Global = R * Vector_Local
        # R should transform Local -> Global.
        # Local Y is "Top" of panel. Local Z is Normal.
        
        # Let's assume at rest:
        # Panel Normal = Z (0,0,1)
        # Panel Top = Y (0,1,0)
        # Panel Right = X (1,0,0)
        
        # We rotate around Z axis by Azimuth (CW from North).
        # If Az=90 (East), we want Y_local to point East.
        # R_az * [0,1,0] = [1,0,0].
        # [ cos -sin 0 ]    [0]   [-sin]
        # [ sin  cos 0 ] *  [1] = [cos ]
        # [  0    0  1 ]    [0]   [ 0  ]
        # If theta = -90 (CCW 90? No CW 90 implies -90). 
        # sin(-90)=-1. [-(-1)] = 1. Correct.
        
        # So Rotation 1 is Rz(-Azimuth).
        
        # Rotation 2: Elevation (Tilt up).
        # We rotate around local X-axis.
        # Angle?
        # Initially Normal is Z. Target is El.
        # Tilt = 90 - El.
        # If El=0 (Horizon), Tilt=90.
        # If El=90 (Zenith), Tilt=0.
        # We rotate around X. Normal (0,0,1) -> (0, -sin(tilt), cos(tilt))??
        # Right hand rule on X: Y -> Z.
        # R_x(tilt):
        # [1 0 0]
        # [0 c -s]
        # [0 s c]
        # Apply to (0,0,1): (0, -s, c).
        # If tilt=90, (0, -1, 0). Points South?
        # We want it to point "forward" (Local Y).
        # Actually in Az-El, we tilt BACK?
        # Let's just use the direct construction. It is safer than guessing Euler conventions.
        
        # Target Normal vector (Global):
        n_x = np.sin(theta_s) * np.sin(phi_s)
        n_y = np.sin(theta_s) * np.cos(phi_s)
        n_z = np.cos(theta_s)
        normal = np.array([n_x, n_y, n_z])

        # Target 'Right' vector (Local X).
        # For Az-El, the pivot axis (Elevation axis) is Horizontal.
        # It is perpendicular to the Azimuth Plane.
        # So Local X is perpendicular to Z-global and Normal.
        # Actually Local X is just in the XY plane, perpendicular to the Azimuth bearing.
        # Bearing = (sin(az), cos(az), 0).
        # Right = (cos(az), -sin(az), 0).
        
        right_x = np.cos(phi_s)
        right_y = -np.sin(phi_s)
        right_z = 0.0
        right = np.array([right_x, right_y, right_z])
        
        # Ensure orthogonality (sun might be at zenith)
        if np.linalg.norm(right) < 1e-6:
             right = np.array([1.0, 0.0, 0.0]) # Default to East

        # Target 'Up' vector (Local Y)
        # Y = Z x X
        up = np.cross(normal, right)
        
        # The Rotation Matrix R = [X_col, Y_col, Z_col]
        # Because R * e1 = X_col.
        R = np.column_stack((right, up, normal))
        
        return R

