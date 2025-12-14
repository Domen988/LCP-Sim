from dataclasses import dataclass

@dataclass
class ScenarioConfig:
    """
    Configuration for the simulation scenario.
    """
    grid_pitch_x: float = 1.05
    grid_pitch_y: float = 1.05
    tolerance: float = 0.02
