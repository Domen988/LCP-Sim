
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
from lcp.core.geometry import PanelGeometry
from lcp.core.config import ScenarioConfig

@dataclass
class SimulationSettings:
    start_date: date = date(2025, 1, 1)
    full_year: bool = False
    duration_days: int = 3
    timestep_min: int = 6

@dataclass
class AppState:
    """Holds the runtime state of the Desktop Application"""
    geometry: PanelGeometry
    config: ScenarioConfig
    sim_settings: SimulationSettings = field(default_factory=SimulationSettings)
    
    # Environment
    sun_az: float = 180.0
    sun_el: float = 45.0
    sun_az: float = 180.0
    sun_el: float = 45.0
    # plant_rotation moved to Config
    
    # Manual Control
    stow_az: float = 0.0
    stow_el: float = 0.0
    
    # Layout (Overrides or cache)
    rows: int = 12
    cols: int = 20
    
    # Persistence
    storage_path: str = "saved_simulations"
    
    # Shared Data
    stow_profile: Optional[object] = None 

    def __post_init__(self):
        from lcp.core.stow import StowProfile
        if self.stow_profile is None:
             self.stow_profile = StowProfile()
