
import streamlit as st
import numpy as np
import os
import time
from datetime import datetime, timedelta

from lcp.core.stow import StowProfile
from lcp.app.visualizer import PlantVisualizer
from lcp.physics.engine import InfiniteKernel
from lcp.core.geometry import PanelGeometry
from lcp.core.config import ScenarioConfig
from lcp.simulation import SimulationRunner

# Constants
STOW_AZ_MIN = -180.0
STOW_AZ_MAX = 180.0 # Changed to +/- 180 per request

def render_stow_recorder(viz_container, geo: PanelGeometry, cfg: ScenarioConfig, sim_results=None):
    st.header("Manual Stow Recorder")

    # --- STATE MANAGEMENT ---
    from lcp.core.solar import SolarCalculator
    solar = SolarCalculator()
    
    if "stow_profile" not in st.session_state:
        st.session_state["stow_profile"] = StowProfile()
        
    if "scrub_time" not in st.session_state:
        st.session_state["scrub_time"] = datetime(2025, 1, 1, 12, 0, 0)
    
    # Init Az/El based on Tracking if not set
    if "stow_az" not in st.session_state:
        # Calculate safe tracking angle
        sun_start = solar.get_position(st.session_state["scrub_time"])
        if sun_start.elevation > 0:
            st.session_state["stow_az"] = sun_start.azimuth
            st.session_state["stow_el"] = sun_start.elevation
        else:
            st.session_state["stow_az"] = 0.0
            st.session_state["stow_el"] = 0.0

    profile: StowProfile = st.session_state["stow_profile"]
    
    # --- HELPER: Resolve Step Size ---
    # Default to 10 if not found in results
    step_minutes = 10
    if sim_results and len(sim_results) > 0:
        # Try to infer from first day's frames
        frames = sim_results[0].get('frames', [])
        if len(frames) >= 2:
            dt1 = frames[0]['time']
            dt2 = frames[1]['time']
            diff = (dt2 - dt1).total_seconds() / 60.0
            step_minutes = int(round(diff))
    
    # --- UI LAYOUT REFACOR: Left (Controls) | Right (Viz) ---
    col_left, col_right = st.columns([1, 2])
    
    # ==========================
    # LEFT COLUMN: CONTROLS
    # ==========================
    with col_left:
        st.subheader("Control Panel")
        
        # 1. PROFILE MGMT
        st.markdown("##### Profile")
        st.text_input("Name", key="prof_name", value=profile.profile_name)
        c_io1, c_io2 = st.columns(2)
        with c_io1:
             if st.button("Load"):
                  uploaded = st.file_uploader("Upload", type=["json"], key="up_json")
                  if uploaded: st.info("Pending integration")
        with c_io2:
             if st.button("Save"):
                  path = os.path.join(os.getcwd(), f"{st.session_state['prof_name']}.json")
                  profile.profile_name = st.session_state['prof_name']
                  profile.save(path)
                  st.success(f"Saved")

        st.divider()

        # 2. TIME CONTROL
        st.markdown("##### Timeline")
        # Step Size Display (Locked if inferred, or editable?)
        # User requested: "taken from the simulation"
        # Let's show it but maybe allow override if needed?
        step_user = st.number_input("Step (min)", 1, 60, step_minutes, key="step_size_user")
        
        # Prev/Next
        c_t1, c_t2, c_t3 = st.columns([1, 2, 1])
        with c_t1:
             if st.button("◀"):
                  st.session_state["scrub_time"] -= timedelta(minutes=step_user)
        with c_t3:
             if st.button("▶"):
                  st.session_state["scrub_time"] += timedelta(minutes=step_user)
        with c_t2:
             st.markdown(f"<div style='text-align:center; font-weight:bold;'>{st.session_state['scrub_time'].strftime('%H:%M')}</div>", unsafe_allow_html=True)
             st.caption(st.session_state["scrub_time"].strftime('%Y-%m-%d'))
             
        st.divider()

        # 3. PLAYBACK
        st.markdown("##### Playback")
        if st.button("▶ Test Profile"):
            play_loop(viz_container, geo, cfg, profile, solar, step_user, None, None)
            
        # CLASH SCENARIO
        st.markdown("##### Clash Analysis")
        # Select Date (if multiple)
        if sim_results:
             dates = [r['summary']['date'] for r in sim_results]
             sel_date = st.selectbox("Date", dates, format_func=lambda d: d.strftime('%Y-%m-%d'))
             
             if st.button("▶ Play Clash Scenario"):
                  # Validate
                  day_res = next((r for r in sim_results if r['summary']['date'] == sel_date), None)
                  if day_res:
                       # Find Clash Frames
                       # Check if 'safety' flag in frames indicates clash?
                       # Usually safety=True means safety MODE used.
                       # We need to check STATES. But states are heavy objects.
                       # Or use 'clash_count' in summary to know IF there are ANY.
                       if day_res['summary']['clash_count'] == 0:
                            st.info("No clashes detected on this day!")
                       else:
                            # Search frames
                            clash_times = []
                            for f in day_res['frames']:
                                 # We need to know if THIS frame had clash.
                                 # The 'safety' flag in frame data (dashboard.py:412) is just Input bool.
                                 # But we have `day_stats["clash_count"]` which aggregates.
                                 # We don't seemingly store per-frame clash boolean in the simplified frame dict?
                                 # Wait, let's check dashboard.py L404.
                                 # It increments count if `safety` (input) is True? 
                                 # No, `safety` variable in loop holds `collision` result from kernel?
                                 # L364: `states, safety = runner.kernel.solve_timestep(...)`
                                 # YES! `safety` variable IS the collision boolean.
                                 # And it is stored in frame dict as "safety".
                                 if f.get('safety', False): 
                                      clash_times.append(f['time'])
                            
                            if not clash_times:
                                 st.warning("Clash count > 0 but no frames marked? Check data.")
                            else:
                                 start_t = clash_times[0] - timedelta(minutes=step_user * 2)
                                 end_t = clash_times[-1] + timedelta(minutes=step_user * 2)
                                 play_loop(viz_container, geo, cfg, profile, solar, step_user, start_t, end_t)
                  else:
                       st.error("Day not found.")
        else:
             st.caption("Run simulation to enable.")

    # ==========================
    # RIGHT COLUMN: VISUALIZER & TEACH PENDANT
    # ==========================
    with col_right:
        # Override Container for Viz
        viz_spot = st.empty()
        
        # SLIDERS & MICRO CONTROLS
        st.subheader("Teach Pendant")
        
        # Logic to Sync Time changes to State
        ensure_state_sync(profile, solar)
        
        c_az, c_el = st.columns(2)
        
        # --- AZIMUTH ---
        with c_az:
             st.caption("Azimuth")
             # Micro Buttons
             ca1, ca2 = st.columns(2)
             if ca1.button("-5°", key="az_m5"): 
                  st.session_state["stow_az"] = max(STOW_AZ_MIN, st.session_state["stow_az"] - 5)
             if ca2.button("+5°", key="az_p5"): 
                  st.session_state["stow_az"] = min(STOW_AZ_MAX, st.session_state["stow_az"] + 5)
             
             az_val = st.slider("Az", STOW_AZ_MIN, STOW_AZ_MAX, st.session_state["stow_az"], 1.0, key="sl_az", label_visibility="collapsed")
             st.session_state["stow_az"] = az_val

        # --- ELEVATION ---
        with c_el:
             st.caption("Elevation")
             limit_ott = st.checkbox("OTT", value=profile.limit_over_the_top, key="chk_ott")
             el_max = 135.0 if limit_ott else 90.0
             
             ce1, ce2 = st.columns(2)
             if ce1.button("-5°", key="el_m5"): 
                  st.session_state["stow_el"] = max(0.0, st.session_state["stow_el"] - 5)
             if ce2.button("+5°", key="el_p5"): 
                  st.session_state["stow_el"] = min(el_max, st.session_state["stow_el"] + 5)
                  
             el_val = st.slider("El", 0.0, el_max, st.session_state["stow_el"], 1.0, key="sl_el", label_visibility="collapsed")
             st.session_state["stow_el"] = el_val

        # ACTION BUTTONS
        c_act1, c_act2 = st.columns(2)
        with c_act1:
             if st.button("Unstow (Sync Tracking)", use_container_width=True):
                  # Calculate tracking angle for current time
                  s_pos = solar.get_position(st.session_state["scrub_time"])
                  st.session_state["stow_az"] = s_pos.azimuth
                  st.session_state["stow_el"] = s_pos.elevation
                  st.rerun()

        with c_act2:
             if st.button("🔴 Record Keyframe", type="primary", use_container_width=True):
                  profile.add_keyframe(st.session_state["scrub_time"], az_val, el_val)
                  st.success("Recorded")
        
        # LIST
        if profile.keyframes:
             with st.expander("Keyframes", expanded=False):
                  for k in profile.keyframes[-5:]:
                       st.text(f"{k.timestamp.strftime('%H:%M')} | {k.az:.1f} / {k.el:.1f}")

        # RENDER STATIC FRAME
        render_frame(viz_spot, geo, cfg, st.session_state["scrub_time"], 
                     st.session_state["stow_az"], st.session_state["stow_el"], solar, "main_static")


def ensure_state_sync(profile, solar):
    if "last_scrub_time" not in st.session_state or st.session_state["last_scrub_time"] != st.session_state["scrub_time"]:
         st.session_state["last_scrub_time"] = st.session_state["scrub_time"]
         # Interpolate or Tracking
         interp = profile.get_position_at(st.session_state["scrub_time"])
         if interp:
             st.session_state["stow_az"] = interp[0]
             st.session_state["stow_el"] = interp[1]
         else:
             s_pos = solar.get_position(st.session_state["scrub_time"])
             if s_pos.elevation > 0:
                 st.session_state["stow_az"] = s_pos.azimuth
                 st.session_state["stow_el"] = s_pos.elevation

def render_frame(container, geo, cfg, time_dt, az, el, solar, key_suffix=""):
    # Physics
    sun_pos = solar.get_position(time_dt)
    local_az = sun_pos.azimuth - 5.0
    override = (az, el)
    
    ker = InfiniteKernel(geo, cfg)
    states, collision = ker.solve_timestep(local_az, sun_pos.elevation, enable_safety=True, stow_override=override)
    
    # Viz
    viz = PlantVisualizer(geo)
    rad_az = np.radians(sun_pos.azimuth)
    rad_el = np.radians(sun_pos.elevation)
    sun_vec = np.array([
        np.sin(rad_az) * np.cos(rad_el),
        np.cos(rad_az) * np.cos(rad_el),
        np.sin(rad_el)
    ])
    fig = viz.render_scene(states, sun_vec, show_pivots=True, show_clash_emphasis=True)
    
    if collision:
         container.error(f"CLASH DETECTED at {time_dt.strftime('%H:%M')}")
    else:
         container.success(f"SAFE at {time_dt.strftime('%H:%M')}")
         
    container.plotly_chart(fig, use_container_width=True, key=f"viz_{key_suffix}")

def play_loop(container, geo, cfg, profile, solar, step_min, start_dt=None, end_dt=None):
    if not profile.keyframes and not start_dt:
        return
        
    if not start_dt:
        start_dt = profile.keyframes[0].timestamp
    if not end_dt:
        end_dt = profile.keyframes[-1].timestamp
        
    curr = start_dt
    status = st.empty()
    
    while curr <= end_dt:
        # Interpolate
        az, el = 0.0, 0.0
        res = profile.get_position_at(curr)
        if res:
            az, el = res
        else:
             # If gap in profile, default to tracking?
             s = solar.get_position(curr)
             az, el = s.azimuth, s.elevation
        
        render_frame(container, geo, cfg, curr, az, el, solar, f"play_{curr.strftime('%H%M')}")
        status.text(f"Playing {curr.strftime('%H:%M')}")
        time.sleep(0.1)
        curr += timedelta(minutes=step_min)
    status.empty()
