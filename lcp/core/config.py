from dataclasses import dataclass

@dataclass
class ScenarioConfig:
    """
    Configuration for the simulation scenario.
    """
    grid_pitch_x: float = 1.7
    grid_pitch_y: float = 1.7
    tolerance: float = 0.05
    total_panels: int = 2400 # 60x40 default
    plant_rotation: float = 5.35 # Clockwise from North
    min_elevation: float = 15.0 # Degrees (User requests min 15°)
    
    # 4x4 Field Configuration
    field_spacing_x: float = 2.3 # Pivot-to-Pivot between fields (X)
    field_spacing_y: float = 2.4 # Pivot-to-Pivot between fields (Y)
    
    # Sun Source
    sun_source: str = "csv" # 'pvlib' or 'csv'
