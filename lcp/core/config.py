from dataclasses import dataclass

@dataclass
class ScenarioConfig:
    """
    Configuration for the simulation scenario.
    """
    grid_pitch_x: float = 1.7
    grid_pitch_y: float = 1.7
    tolerance: float = 0.03
    grid_pitch_y: float = 1.7
    tolerance: float = 0.03
    total_panels: int = 2400 # 60x40 default
    plant_rotation: float = 5.0 # Clockwise from North
