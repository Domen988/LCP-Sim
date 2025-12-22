
import os
import streamlit.components.v1 as components

# Declare component
_component_func = components.declare_component(
    "three_viz",
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)))
)

def three_viz(geometry, scene_state, playback_frames=None, initial_stow=None, key=None):
    """
    Renders the Three.js Visualizer.
    
    Args:
        geometry (dict): {width, length, rows, cols, pitch_x, pitch_y, thickness, offset_z}
        scene_state (dict): {sun_az, sun_el, plant_rotation}
        playback_frames (list): Optional list of frames for animation [{time_str, az, el, sun_az, sun_el}, ...]
        initial_stow (dict): {az, el} for initial slider position
    
    Returns:
        dict: {az, el} when "Record" is clicked, or None if no event.
    """
    
    component_value = _component_func(
        geometry=geometry,
        scene_state=scene_state,
        playback_frames=playback_frames,
        initial_stow=initial_stow,
        key=key,
        default=None,
        height=600 # Fix blinking
    )
    
    return component_value
