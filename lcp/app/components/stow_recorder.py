
import streamlit as st
import numpy as np
import os
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go

from lcp.core.stow import StowProfile
from lcp.app.visualizer import PlantVisualizer
from lcp.physics.engine import InfiniteKernel
from lcp.core.geometry import PanelGeometry
from lcp.core.config import ScenarioConfig
from lcp.core.solar import SolarCalculator

# Use our new JS Visualizer
from lcp.app.components.three_viz import three_viz

# Constants
PLANT_ROTATION = 5.0 # Ensure this matches dashboard.py

def render_stow_recorder(viz_container, geo: PanelGeometry, cfg: ScenarioConfig, sim_results=None):
    # Compact Header
    c_head, c_info = st.columns([2, 1])
    with c_head:
        st.subheader("Teach Mode (High Speed)")
    
    # --- STATE MANAGEMENT ---
    solar = SolarCalculator()
    
    if "stow_profile" not in st.session_state:
        st.session_state["stow_profile"] = StowProfile()
        
    if "scrub_time" not in st.session_state:
        st.session_state["scrub_time"] = datetime(2025, 1, 1, 12, 0, 0)
    
    # We still keep these in session state for saving/loading, 
    # but updates come from the JS component now.
    if "stow_az" not in st.session_state: st.session_state["stow_az"] = 0.0
    if "stow_el" not in st.session_state: st.session_state["stow_el"] = 0.0

    profile: StowProfile = st.session_state["stow_profile"]
    
    # Step Size Logic
    step_minutes = 10
    if sim_results and len(sim_results) > 0:
        frames = sim_results[0].get('frames', [])
        if len(frames) >= 2:
            step_minutes = int(round((frames[1]['time'] - frames[0]['time']).total_seconds() / 60.0))

    # --- LAYOUT: Left (Controls) [1] | Right (Viz) [3] ---
    col_left, col_right = st.columns([1, 2.5])
    
    # ==========================
    # LEFT: CONTROL PANEL
    # ==========================
    with col_left:
        # 1. TIMELINE
        st.caption("**Timeline**")
        c_time1, c_time2 = st.columns([1, 1])
        with c_time1:
             if st.button("◀", use_container_width=True):
                  st.session_state["scrub_time"] -= timedelta(minutes=step_minutes)
        with c_time2:
             if st.button("▶", use_container_width=True):
                  st.session_state["scrub_time"] += timedelta(minutes=step_minutes)
        
        # Date/Time Display
        dt_str = st.session_state["scrub_time"].strftime('%Y-%m-%d\n%H:%M')
        st.code(dt_str)

        st.divider()

        # 2. FILE
        st.caption("**Profile**")
        st.text_input("Name", key="prof_name", value=profile.profile_name, label_visibility="collapsed")
        if st.button("💾 Save", use_container_width=True):
             path = os.path.join(os.getcwd(), f"{st.session_state['prof_name']}.json")
             profile.profile_name = st.session_state['prof_name']
             profile.save(path)
             st.success("Saved")
             
        st.divider()
        
        # 3. CONTEXT SELECTOR (Play Day or Play Clash)
        st.caption("**Animation**")
        # Instead of buttons triggering a loop, they will set mode for the React component
        
        active_mode = "MANUAL"
        playback_data = None
        
        # We use radio or buttons to switch what is sent to the component
        anim_mode = st.radio("Mode", ["Manual", "Play Day", "Play Clash"], label_visibility="collapsed")
        
        if anim_mode == "Play Day":
             active_mode = "PLAYBACK"
             # Generate Full Day Frames
             playback_data = generate_playback_frames(st.session_state["scrub_time"].date(), 
                                                      step_minutes, geo, cfg, profile, solar, "DAY")
                                                      
        elif anim_mode == "Play Clash":
             active_mode = "PLAYBACK"
             if sim_results:
                  # Use selected date logic
                  dates = [r['summary']['date'] for r in sim_results]
                  # Try to match current scrub date
                  curr_d = st.session_state["scrub_time"].date()
                  if curr_d in dates:
                       playback_data = generate_playback_frames(curr_d, step_minutes, geo, cfg, profile, solar, "CLASH", sim_results)
                  else:
                       st.warning("No sim data for current date.")
             else:
                  st.warning("No sim results loaded.")


    # ==========================
    # RIGHT: JS VIZ
    # ==========================
    with col_right:
        # Prepare Data for Component
        
        # 1. Geometry Dict
        geo_dict = {
             "rows": 12, # Hardcoded fallback if not in cfg? 
             # No, cfg has `total_panels` but not rows/cols strictly?
             # dashboard.py uses n_rows, n_cols. But `render_stow_recorder` doesn't receive them explicitly?
             # `cfg` contains total_panels, grid_pitch_x...
             # WAITING: `render_stow_recorder` signature doesn't pass n_rows/n_cols.
             # We need to assume/extract or update signature. 
             # For now, let's look at `dashboard.py`: calls with `current_geo, current_cfg`.
             # `current_cfg` handles pitches. `current_geo` handles width/len.
             # Rows/Cols are stuck in dashboard local vars.
             # HACK: Infer from total_panels assuming standard aspect? No.
             # We should update dashboard to pass rows/cols or put in cfg.
             # Let's assume standard 10x10 or extract from cfg if possible? 
             # `cfg` is ScenarioConfig object.
             # Let's use defaults 12x20 if missing, or maybe they are in session_state from sidebar?
             "rows": 12, 
             "cols": 20, 
             "width": geo.width,
             "length": geo.length,
             "thickness": geo.thickness,
             "pitch_x": cfg.grid_pitch_x,
             "pitch_y": cfg.grid_pitch_y, 
             "offset_z": geo.pivot_offset[2]
        }
        
        # 2. Scene State
        s = solar.get_position(st.session_state["scrub_time"])
        scene_state = {
             "sun_az": s.azimuth,
             "sun_el": s.elevation,
             "plant_rotation": PLANT_ROTATION,
             "current_time_str": st.session_state["scrub_time"].strftime('%H:%M')
        }
        
        # 3. Initial Stow (for Slider Init)
        # Interpolate current profile pose
        i_az, i_el = 0.0, 0.0
        interp = profile.get_position_at(st.session_state["scrub_time"])
        if interp:
             i_az, i_el = interp
        else:
             i_az, i_el = s.azimuth, s.elevation
             
        initial_stow = {"az": i_az, "el": i_el}
        
        # 4. RENDER COMPONENT
        # We listen for return value. 
        # The return value is the "Recorded" state.
        
        rec_val = three_viz(
             geometry=geo_dict,
             scene_state=scene_state,
             playback_frames=playback_data,
             initial_stow=initial_stow,
             key="three_viz_inst"
        )
        
        # Handle Record Event
        if rec_val:
             # User pressed Record in JS
             az = rec_val['az']
             el = rec_val['el']
             profile.add_keyframe(st.session_state["scrub_time"], az, el)
             st.toast(f"Recorded: Ax={az:.1f}, El={el:.1f}")

def generate_playback_frames(date_obj, step, geo, cfg, profile, solar, mode, sim_results=None):
    # Determine range
    start_dt = datetime.combine(date_obj, datetime.min.time()) + timedelta(hours=6)
    end_dt = datetime.combine(date_obj, datetime.min.time()) + timedelta(hours=20)
    
    if mode == "CLASH" and sim_results:
         # Find clash range
         day = next((r for r in sim_results if r['summary']['date'] == date_obj), None)
         if day and day['summary']['clash_count'] > 0:
              clashes = [f['time'] for f in day['frames'] if f.get('safety')]
              if clashes:
                   start_dt = clashes[0] - timedelta(minutes=step*3)
                   end_dt = clashes[-1] + timedelta(minutes=step*3)
         else:
              return [] # No clash
              
    # Generate
    frames = []
    curr = start_dt
    while curr <= end_dt:
         s = solar.get_position(curr)
         
         # Interpolate
         az, el = 0.0, 0.0
         pos = profile.get_position_at(curr)
         if pos: az, el = pos
         else: az, el = s.azimuth, s.elevation
         
         frames.append({
              "time_str": curr.strftime('%H:%M'),
              "az": az,
              "el": el,
              "sun_az": s.azimuth,
              "sun_el": s.elevation
         })
         curr += timedelta(minutes=step)
         
    return frames
