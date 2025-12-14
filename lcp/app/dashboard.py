import streamlit as st
import sys
import os

# --- PATH SETUP ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
import time

# --- MODULE IMPORTS ---
from lcp.core.geometry import PanelGeometry
from lcp.core.config import ScenarioConfig
from lcp.simulation import SimulationRunner
from lcp.app.visualizer import PlantVisualizer

PLANT_ROTATION = 5.0

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(page_title="LCP-Sim v3.0", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem; 
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    h1, h2, h3 {
        margin-top: 0rem;
        margin-bottom: 0.5rem;
    }
    /* Hide Streamlit Header */
    .stApp > header {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR CONTROL ROOM
# ==========================================
st.sidebar.title("LCP Control Room")
view_mode = st.sidebar.radio("View", ["Simulation Results", "3D Analysis"], index=1)
st.sidebar.markdown("---")

# Geometry Config
with st.sidebar.expander("Geometry Config", expanded=False):
    panel_width = st.number_input("Panel Width (m)", 0.5, 5.0, 1.0, 0.1)
    panel_length = st.number_input("Panel Length (m)", 0.5, 5.0, 1.0, 0.1)
    pivot_depth = st.number_input("Pivot Depth from Glass (m)", 0.0, 0.5, 0.05, 0.01)

    # Use number_input for pitch to return float directly
    pitch_x = st.number_input("Pitch X (m)", 0.5, 10.0, 1.05, 0.01)
    pitch_y = st.number_input("Pitch Y (m)", 0.5, 10.0, 1.05, 0.01)
    tolerance = st.number_input("Clash Tolerance (m)", 0.0, 0.5, 0.02, 0.01)

# Plant Size
with st.sidebar.expander("Plant Sizing", expanded=False):
    n_rows = st.number_input("Rows", 1, 1000, 20)
    n_cols = st.number_input("Cols", 1, 1000, 12)

# Simulation Settings
with st.sidebar.expander("Simulation Settings", expanded=False):
    # Fix: Use date() object for default
    start_date = st.date_input("Start Date", date(2025, 1, 1))
    
    full_year_sim = st.checkbox("Full Year Simulation (365 Days)", value=False)
    
    if full_year_sim:
        sim_days = 365
        # Show disabled input for feedback
        st.number_input("Duration (Days)", value=365, disabled=True, key="sim_days_disp")
    else:
        sim_days = st.number_input("Duration (Days)", 1, 3650, 3) 

# Run Button
if st.sidebar.button("Run Simulation", type="primary"):
    st.session_state["run_trigger"] = True
    st.session_state["sim_timestamp"] = datetime.now() # Force refresh

# ==========================================
# 3. SIMULATION LOGIC
# ==========================================

@st.cache_resource
def get_sim_runner_v3():
    # Ensure csv path is correct relative to execution
    r = SimulationRunner("Koster direct normal irradiance_Wh per square meter.csv")
    r.load_data() # Pre-calculate Splines to ensure they are cached with the object
    return r

runner = get_sim_runner_v3()
data_matrix = runner.load_data()

# Initialize Session State
if "simulation_results" not in st.session_state:
    st.session_state["simulation_results"] = None

# --- RUN LOOP ---
if st.session_state.get("run_trigger", False):
    with st.spinner("Running Simulation..."):
        # 1. Setup Geometry
        thickness = 0.05
        off_z = pivot_depth - (thickness / 2.0) # Pos Z: Glass Above Pivot
        
        geo = PanelGeometry(
            width=panel_width,
            length=panel_length,
            thickness=thickness,
            pivot_offset=(0.0, 0.0, off_z)
        )
        cfg = ScenarioConfig(
            grid_pitch_x=pitch_x,
            grid_pitch_y=pitch_y, 
            tolerance=tolerance
        )
        
        # 2. Update Kernel (3x3 RVE)
        runner.kernel.geo = geo
        runner.kernel.cfg = cfg
        runner.kernel.collider.geo = geo
        runner.kernel.collider.cfg = cfg
        runner.kernel.pivots = {}
        
        for r in range(3):
            for c in range(3):
                # Standard grid generation
                y = (r - 1) * cfg.grid_pitch_y
                x = (c - 1) * cfg.grid_pitch_x
                runner.kernel.pivots[(r,c)] = np.array([float(x), float(y), 0.0])
        
        # 3. Simulation Loop
        scale_efficiency = (n_rows * n_cols) / 9.0 
        all_days = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Calculate total steps for progress bar
        steps_per_day = 240 # 24h / 6min
        total_steps = sim_days * steps_per_day
        global_step = 0
        
        PLANT_ROTATION = 5.0 # Constant definition
        
        for d in range(sim_days):
            current_day_date = start_date + timedelta(days=d)
            start_dt = datetime.combine(current_day_date, datetime.min.time())
            # Generate 6-min intervals
            steps = [start_dt + timedelta(minutes=6*i) for i in range(steps_per_day)]
            
            day_frames = []
            day_stats = {
                "date": current_day_date,
                "theo_kwh": 0.0, "act_kwh": 0.0,
                "stow_loss_kwh": 0.0, "shad_loss_kwh": 0.0,
                "clash_count": 0
            }
            
            for i, dt in enumerate(steps):
                global_step += 1
                if i % 20 == 0: # UI Update throttle
                    progress_bar.progress(min(1.0, global_step / total_steps))
                    status_text.text(f"Simulating {current_day_date} | {dt.strftime('%H:%M')}")
                
                # A. Physics
                sun = runner.solar.get_position(dt)
                local_az = sun.azimuth - PLANT_ROTATION
                
                # B. Solve State
                states, safety = runner.kernel.solve_timestep(local_az, sun.elevation, enable_safety=True)
                dni = runner.get_dni(dt, data_matrix)
                
                # C. Calculate Power
                panel_area = geo.width * geo.length
                
                # Theoretical (Ideal Tracking)
                step_theo_w_rve = (9.0 * panel_area) * dni
                
                # Actual
                step_act_w_rve = sum([panel_area * dni * s.power_factor for s in states])
                
                # Losses
                step_stow_loss_w = 0.0
                step_shad_loss_w = 0.0
                
                for s in states:
                    p_theo_panel = panel_area * dni
                    if s.mode == "STOW":
                        step_stow_loss_w += p_theo_panel
                    else:
                        step_shad_loss_w += p_theo_panel * s.shadow_loss
                        
                # Extrapolate to Plant
                day_stats["theo_kwh"] += (step_theo_w_rve * scale_efficiency) * 0.1 / 1000.0
                day_stats["act_kwh"] += (step_act_w_rve * scale_efficiency) * 0.1 / 1000.0
                day_stats["stow_loss_kwh"] += (step_stow_loss_w * scale_efficiency) * 0.1 / 1000.0
                day_stats["shad_loss_kwh"] += (step_shad_loss_w * scale_efficiency) * 0.1 / 1000.0
                
                if safety:
                    day_stats["clash_count"] += 1
                
                # D. Save Frame
                day_frames.append({
                    "time": dt,
                    "sun_az": sun.azimuth,
                    "sun_el": sun.elevation,
                    "safety": safety,
                    "act_w": step_act_w_rve * scale_efficiency,
                    "theo_w": step_theo_w_rve * scale_efficiency,
                    "states": states 
                })
            
            all_days.append({
                "summary": day_stats,
                "frames": day_frames,
                "geo": geo,
                "cfg": cfg # FIX: Save the Config used for this run!
            })
            
        progress_bar.progress(1.0)
        status_text.success("Complete")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
            
        st.session_state["simulation_results"] = all_days
    st.session_state["run_trigger"] = False


# ==========================================
# 4. RESULTS DASHBOARD
# ==========================================
# tab1, tab2 = st.tabs(["Simulation Results", "3D Analysis"]) -> Removed for Sidebar Nav

results = st.session_state.get("simulation_results")

if not results:
    st.info("👈 Please configure Geometry and Click 'Run Simulation'.")
    st.stop()

# --- VIEW 1: METRICS ---
if view_mode == "Simulation Results":
    
    col_left, col_right = st.columns([1, 2])
    
    # 1. AGGREGATE STATS
    cum_theo = sum([d['summary']['theo_kwh'] for d in results])
    cum_act = sum([d['summary']['act_kwh'] for d in results])
    cum_stow = sum([d['summary']['stow_loss_kwh'] for d in results])
    cum_shad = sum([d['summary']['shad_loss_kwh'] for d in results])
    
    # Derivations
    total_loss = cum_theo - cum_act
    # Whatever isn't explained by Stow or Shadow is "Other" (Cosine, Limits)
    cum_other = total_loss - cum_stow - cum_shad
    
    eff_pct = (cum_act / cum_theo * 100) if cum_theo > 0 else 0
    stow_pct = (cum_stow / cum_theo * 100) if cum_theo > 0 else 0
    shad_pct = (cum_shad / cum_theo * 100) if cum_theo > 0 else 0
    other_pct = (cum_other / cum_theo * 100) if cum_theo > 0 else 0
    
    # --- LEFT COLUMN: STATS & TABLE ---
    # --- LEFT COLUMN: STATS & TABLE ---
    with col_left:
        st.subheader("Performance Summary")
        
        summ_html = f"""
<div style="font-size:0.95em; line-height:1.6; border:1px solid rgba(250,250,250,0.2); padding:15px; border-radius:8px; margin-bottom:20px;">
    <div style="display:grid; grid-template-columns: 1.5fr 1fr; row-gap:8px;">
        <div style="color:#aaa;">Total Theoretical</div> <div style="font-weight:bold; text-align:right;">{cum_theo/1000:,.2f} MWh</div>
        <div style="color:#aaa;">Total Actual</div>      <div style="font-weight:bold; text-align:right;">{cum_act/1000:,.2f} MWh</div>
        <div style="border-bottom:1px solid rgba(250,250,250,0.2); grid-column:1/-1; margin:4px 0;"></div>
        <div style="color:#aaa;">Efficiency</div>        <div style="font-weight:bold; text-align:right; color:#4CAF50;">{eff_pct:.1f}%</div>
        <div style="color:#aaa;">Stow Loss</div>         <div style="font-weight:bold; text-align:right; color:#FF5252;">{stow_pct:.1f}%</div>
        <div style="color:#aaa;">Shadow Loss</div>       <div style="font-weight:bold; text-align:right; color:#FFC107;">{shad_pct:.2f}%</div>
        <div style="color:#aaa;">Other / Cosine</div>    <div style="font-weight:bold; text-align:right; color:#9E9E9E;">{other_pct:.2f}%</div>
    </div>
</div>
"""
        st.markdown(summ_html, unsafe_allow_html=True)
        
        st.subheader("Daily Data")
        rows = []
        for res in results:
            s = res['summary']
            t = s['theo_kwh']
            a = s['act_kwh']
            rows.append({
                "Date": s['date'].strftime("%Y-%m-%d"),
                "Theo [kWh]": f"{t:.1f}",
                "Act [%]": f"{(a/t)*100:.1f}" if t > 0 else "0",
                "Stow [%]": f"{(s['stow_loss_kwh']/t)*100:.1f}" if t > 0 else "0"
            })
        
        df_table = pd.DataFrame(rows)
        # Display Table
        st.dataframe(df_table, use_container_width=True, hide_index=True, height=500)

    # --- RIGHT COLUMN: SURFACE PLOT ---
    with col_right:
        st.subheader("Annual Power Landscape")
        
        # Prepare Data Matrix (X=Time, Y=Day, Z=Power)
        # Filter 05:00 - 20:00
        z_data = []
        y_dates = []
        x_times = None 
        
        for day in results:
            # Filter frames for 05:00 to 20:00 (Inclusive of hour 20)
            valid_frames = [f for f in day['frames'] if 5 <= f['time'].hour <= 20]
            
            # Extract Power
            row = [f['act_w']/1000.0 for f in valid_frames]
            z_data.append(row)
            y_dates.append(day['summary']['date'].strftime("%Y-%m-%d"))
            
            if x_times is None and valid_frames:
                x_times = [f['time'].strftime("%H:%M") for f in valid_frames]
        
        if z_data and x_times:
            fig_surf = go.Figure()
            
            # 1. Surface (Actual)
            fig_surf.add_trace(go.Surface(
                z=z_data, x=x_times, y=y_dates,
                colorscale='Viridis', 
                colorbar=dict(title="Power (kW)", len=0.6, y=0.5)
            ))
            
            # 2. Theoretical Lines (Wireframe)
            # Create a single trace with disjoint lines (separated by None)
            wf_x, wf_y, wf_z = [], [], []
            
            for day in results:
                valid_frames = [f for f in day['frames'] if 5 <= f['time'].hour <= 20]
                if valid_frames:
                    # X: Times
                    ts = [f['time'].strftime("%H:%M") for f in valid_frames]
                    # Y: Date (Repeated)
                    d_str = day['summary']['date'].strftime("%Y-%m-%d")
                    ds = [d_str] * len(ts)
                    # Z: Theo
                    vs = [f['theo_w']/1000.0 for f in valid_frames]
                    
                    wf_x.extend(ts)
                    wf_y.extend(ds)
                    wf_z.extend(vs)
                    
                    # Add Break (None) to separate lines
                    wf_x.append(None)
                    wf_y.append(None)
                    wf_z.append(None)

            if wf_x:
                fig_surf.add_trace(go.Scatter3d(
                    x=wf_x, y=wf_y, z=wf_z,
                    mode='lines', name='Theoretical',
                    line=dict(color='black', width=3, dash='dot'),
                    connectgaps=False
                ))
            
            fig_surf.update_layout(
                scene=dict(
                    xaxis=dict(title="Time (Morning → Evening)"),
                    yaxis=dict(title="Date"),
                    zaxis=dict(title="Power (kW)"),
                    aspectmode='manual', 
                    aspectratio=dict(x=1.0, y=1.0, z=1.0),
                    camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.5))
                ),
                margin=dict(l=0, r=0, b=0, t=10), # No Title
                height=600, 
                legend=dict(x=0, y=1)
            )
            st.plotly_chart(fig_surf, use_container_width=True)
        else:
            st.warning("Insufficient data to generate Surface Plot.")

# --- VIEW 2: 3D ANALYSIS ---
elif view_mode == "3D Analysis":
    
    # 3-Column Layout: [Table (1)] | [Info & Controls (1)] | [Viz & Charts (2)]
    
    # 3-Column Layout: [Table (1)] | [Info & Controls (1)] | [Viz & Charts (2)]
    col_left, col_mid, col_right = st.columns([1, 1, 2])
    
    # --- LEFT: SELECTION ---
    with col_left:
        st.markdown("### Results")
        
        # Session State for Selection
        if 'day_selection_idx' not in st.session_state:
            st.session_state['day_selection_idx'] = 0
            
        rows_3d = []
        for i, r in enumerate(results):
            s = r['summary']
            t_val = s['theo_kwh']
            a_val = s['act_kwh']
            stow_val = s['stow_loss_kwh']
            rows_3d.append({
                "Date": s['date'].strftime("%Y-%m-%d"),
                "Theo": f"{t_val:.1f}",
                "Act %": f"{(a_val/t_val)*100:.1f}" if t_val > 0 else "0",
                "Stow %": f"{(stow_val/t_val)*100:.1f}" if t_val > 0 else "0",
            })
        df_3d = pd.DataFrame(rows_3d)
        
        # Selection
        current_idx = st.session_state['day_selection_idx']
        # Selection
        current_idx = st.session_state['day_selection_idx']
        
        event = st.dataframe(
            df_3d, 
            use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row",
            height=550
        )
        
        # Update Selection
        if event.selection and event.selection.rows:
            st.session_state['day_selection_idx'] = event.selection.rows[0]
        
        sel_idx_3d = st.session_state['day_selection_idx']

    # --- DATA PREP ---
    day_data = results[sel_idx_3d]
    geo_used = day_data['geo']
    cfg_used = day_data['cfg']
    day_frames_light = [f for f in day_data['frames'] if f['sun_el'] > 0]
    
    if not day_frames_light:
        st.warning("No daylight data for selected day.")
        st.stop()
        
    times_str = [f['time'].strftime("%H:%M") for f in day_frames_light]
    viz = PlantVisualizer(geo_used)
    
    # Ghost Box
    p_x = cfg_used.grid_pitch_x
    p_y = cfg_used.grid_pitch_y
    v_len = geo_used.length
    limit_x = 3.0 * p_x * 0.7
    limit_y = 3.0 * p_y * 0.7
    limit_z = 0.8 * v_len
    ghost_trace = go.Scatter3d(
        x=[-limit_x, limit_x, -limit_x, limit_x, 0, 0],
        y=[-limit_y, -limit_y, limit_y, limit_y, 0, 0],
        z=[-limit_z, -limit_z, -limit_z, -limit_z, limit_z, -limit_z],
        mode='markers', marker=dict(opacity=0), showlegend=False, hoverinfo='skip'
    )
    
    scene_cfg = dict(
        xaxis=dict(visible=True, showgrid=True, title="E"),
        yaxis=dict(visible=True, showgrid=True, title="N"),
        zaxis=dict(visible=True, showgrid=True, title="H"),
        aspectmode='data',
        camera=dict(projection=dict(type="orthographic"))
    )

    # --- MID: CONTROLS & INFO PLACEHOLDER ---
    with col_mid:
        st.markdown("### Frame Info")
        ph_frame_info = st.empty()
        
        st.markdown("### Controls")
        enable_anim = st.checkbox("Smooth Animation", value=False)
        show_rays = st.checkbox("Show Sunrays", value=True)
        show_pivots = st.checkbox("Show Pivots", value=True)
        show_stow = st.checkbox("Stow", value=True, help="Uncheck to see collisions (Safety Off)")

    # --- RIGHT: VIZ & SLIDER & CHART ---
    with col_right:
        ph_viz_cont = st.container() # Top: Viz
        ph_slider = st.empty()       # Mid: Slider
        ph_chart = st.empty()        # Bot: Power Chart
        ph_sun_chart = st.empty()    # Bot: Sun Chart
        
    # --- SHARED CHART GENERATION ---
    y_act = [f['act_w']/1000.0 for f in day_frames_light]
    y_theo = [f['theo_w']/1000.0 for f in day_frames_light]
    x_times = times_str
    clash_x = [f['time'].strftime("%H:%M") for f in day_frames_light if f['safety']]
    clash_y = [f['act_w']/1000.0 for f in day_frames_light if f['safety']]
    
    # 1. POWER CHART
    mc = go.Figure()
    mc.add_trace(go.Scatter(x=x_times, y=y_theo, name="Theo [kW]", line=dict(color='gray', dash='dot')))
    mc.add_trace(go.Scatter(x=x_times, y=y_act, name="Actual [kW]", fill='tozeroy', line=dict(color='#1f77b4')))
    
    if clash_x:
        mc.add_trace(go.Scatter(x=clash_x, y=clash_y, mode='markers', name="Clash", marker=dict(color='red', size=8, symbol='x')))
    
    # Removed yaxis title to align chart with slider
    mc.update_layout(height=250, margin=dict(l=0,r=0,t=20,b=20), xaxis=dict(title="Time"), yaxis=dict(title=None), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # 2. SUN PROFILE CHART
    y_az = [f['sun_az'] for f in day_frames_light]
    y_el = [f['sun_el'] for f in day_frames_light]
    
    fig_sun = go.Figure()
    fig_sun.add_trace(go.Scatter(x=x_times, y=y_az, name="Azimuth [°]", line=dict(color='orange')))
    fig_sun.add_trace(go.Scatter(x=x_times, y=y_el, name="Elevation [°]", line=dict(color='gold'), yaxis="y2"))
    
    fig_sun.update_layout(
        height=200, margin=dict(l=0,r=0,t=20,b=20),
        xaxis=dict(title="Time"),
        yaxis=dict(title="Azimuth", side="left", title_standoff=0),
        yaxis2=dict(title="Elevation", side="right", overlaying="y", range=[0, 90], title_standoff=0),
        showlegend=True,
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center")
    )

    # --- SHARED INFO HELPER ---
    def get_info_html(frame, day_sum):
        act_p = (frame['act_w']/frame['theo_w'])*100 if frame['theo_w']>0 else 0
        safe_html = "<span style='color:green'>✅ SAFE</span>" if not frame['safety'] else "<span style='color:red'>⚠️ CLASH</span>"
        return f"""
        <div style="font-size:0.9em; display: grid; grid-template-columns: 80px 1fr; row-gap: 4px; margin-bottom: 1rem;">
            <div style="color:gray">Date</div>      <div style="font-weight:bold">{day_sum['date'].strftime("%Y-%m-%d")}</div>
            <div style="color:gray">Time</div>      <div style="font-weight:bold">{frame['time'].strftime("%H:%M")}</div>
            <div style="color:gray">Sun Az</div>    <div>{frame['sun_az']:.1f}°</div>
            <div style="color:gray">Sun El</div>    <div>{frame['sun_el']:.1f}°</div>
            <div style="color:gray">Total Pwr</div> <div>{frame['theo_w']/1000:.1f} kW</div>
            <div style="color:gray">Actual %</div>  <div>{frame['act_w']/1000:.1f} kW ({act_p:.0f}%)</div>
            <div style="color:gray">Status</div>    <div>{safe_html}</div>
        </div>
        """

    # --- RENDER LOGIC ---
    if enable_anim:
        # ANIMATION MODE
        ph_slider.empty() # Hide slider
        
        # 1. Render Charts (Static)
        with ph_chart:
            st.plotly_chart(mc, use_container_width=True)
        with ph_sun_chart:
            st.plotly_chart(fig_sun, use_container_width=True)
            
        # 2. Render Info (Initial Frame)
        with ph_frame_info:
            st.markdown(get_info_html(day_frames_light[0], day_data['summary']), unsafe_allow_html=True)
            st.caption("(Values update in 3D View during animation)")
        
        # 3. Render Viz (Animation)
        with ph_viz_cont:
            prog_bar = st.progress(0, text="Initializing Animation...")
            
        frames = []
        static_traces = viz._get_static_traces() + [ghost_trace]
        
        # Pre-Init Kernel
        if not show_stow:
             runner.kernel.geo = geo_used
             runner.kernel.collider.geo = geo_used
             runner.kernel.pivots = {}
             for r in range(3):
                for c in range(3):
                    y = (r - 1) * p_y
                    x = (c - 1) * p_x
                    runner.kernel.pivots[(r,c)] = np.array([float(x), float(y), 0.0])

        count = len(day_frames_light)
        for i, f in enumerate(day_frames_light):
            if i % 5 == 0: prog_bar.progress((i+1)/count, text=f"Rendering Frame {i+1}/{count}...")
            
            states = f['states']
            if not show_stow:
                    az = f['sun_az'] - PLANT_ROTATION
                    states, _ = runner.kernel.solve_timestep(az, f['sun_el'], enable_safety=False)
            
            az_rad = np.radians(f['sun_az'] - PLANT_ROTATION)
            el_rad = np.radians(f['sun_el'])
            sv = np.array([np.cos(el_rad)*np.sin(az_rad), np.cos(el_rad)*np.cos(az_rad), np.sin(el_rad)])
            
            dyn = viz._get_dynamic_traces(states, sv, show_rays=show_rays, show_pivots=show_pivots)
            
            safe_txt = "⚠️ CLASH" if f['safety'] else "✅ SAFE"
            act_pct = (f['act_w']/f['theo_w'])*100 if f['theo_w'] > 0 else 0
            
            ft_text = f"<b>{f['time'].strftime('%H:%M')}</b><br>Sun: {f['sun_az']:.1f}°/{f['sun_el']:.1f}°<br>{f['act_w']/1000:.1f}kW ({act_pct:.0f}%)<br>{safe_txt}"
            
            frames.append(go.Frame(
                data=static_traces + dyn, 
                name=f['time'].strftime("%H:%M"),
                layout=dict(annotations=[dict(
                    text=ft_text, x=0, y=1, xref="paper", yref="paper", showarrow=False, align="left", bgcolor="rgba(255,255,255,0.7)"
                )])
            ))
        
        prog_bar.progress(1.0, text="Transferring to Client...")
        time.sleep(0.5)
        
        fig = go.Figure(data=frames[0].data, frames=frames)
        fig.update_layout(
            updatemenus=[dict(
                type='buttons', showactive=False, x=0, y=-0.15, xanchor='left', direction='left',
                buttons=[dict(label='▶', method='animate', args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
                         dict(label='⏸', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate')])])],
            sliders=[dict(currentvalue=dict(prefix=""), pad=dict(t=50), len=0.9, x=0.1, steps=[dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode='immediate')], label=f.name, method='animate') for f in frames])],
            margin=dict(l=0,r=0,b=60,t=40), height=400, scene=scene_cfg, uirevision='sim_3d_anim'
        )
        fig.update_layout(annotations=frames[0].layout.annotations)
        
        prog_bar.empty()
        
        with ph_viz_cont:
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        # STATIC MODE
        # 1. Slider
        sel_time_str = None
        start_t = times_str[len(times_str)//2]
        if 'viz_time_slider' not in st.session_state: st.session_state['viz_time_slider'] = start_t
        with ph_slider:
             sel_time_str = st.select_slider(
                "Time Step", options=times_str, value=st.session_state['viz_time_slider'], 
                label_visibility="collapsed", key='viz_time_slider'
            )

        # 2. Logic (Get Selected Frame)
        t_idx = times_str.index(sel_time_str)
        frame = day_frames_light[t_idx]
        
        # Calc
        states = frame['states']
        if not show_stow:
             runner.kernel.geo = geo_used
             runner.kernel.collider.geo = geo_used
             runner.kernel.pivots = {}
             for r in range(3):
                    for c in range(3):
                        y = (r - 1) * p_y
                        x = (c - 1) * p_x
                        runner.kernel.pivots[(r,c)] = np.array([float(x), float(y), 0.0])
             local_az = frame['sun_az'] - PLANT_ROTATION 
             states, _ = runner.kernel.solve_timestep(local_az, frame['sun_el'], enable_safety=False)
        
        az_rad = np.radians(frame['sun_az'] - PLANT_ROTATION)
        el_rad = np.radians(frame['sun_el'])
        sv = np.array([np.cos(el_rad)*np.sin(az_rad), np.cos(el_rad)*np.cos(az_rad), np.sin(el_rad)])
        
        # 3. Viz
        fig = viz.render_scene(states, sv, show_rays=show_rays, show_pivots=show_pivots)
        fig.add_trace(ghost_trace)
        fig.update_layout(margin=dict(l=0,r=0,b=0,t=0), height=400, uirevision='sim_3d_view', scene=scene_cfg)
        
        with ph_viz_cont:
            st.plotly_chart(fig, use_container_width=True)
            
        # 4. Chart (Add marker)
        if sel_time_str:
             mc.add_vline(x=sel_time_str, line=dict(color="black", dash="dot"))
             fig_sun.add_vline(x=sel_time_str, line=dict(color="black", dash="dot"))
        
        with ph_chart:
            st.plotly_chart(mc, use_container_width=True)
        with ph_sun_chart:
            st.plotly_chart(fig_sun, use_container_width=True)
            
        # 5. Info
        with ph_frame_info:
            st.markdown(get_info_html(frame, day_data['summary']), unsafe_allow_html=True)
