import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import json

# -----------------------------------------------------------------------------
# 1. State & Config
# -----------------------------------------------------------------------------
# Must be first
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "LIBRARY" # LIBRARY | SINGLE | COMPARE
if "sim1_path" not in st.session_state:
    st.session_state.sim1_path = None
if "sim1_name" not in st.session_state:
    st.session_state.sim1_name = None
if "sim2_path" not in st.session_state:
    st.session_state.sim2_path = None
if "sim2_name" not in st.session_state:
    st.session_state.sim2_name = None

# Collapse sidebar if in Compare mode to maximise space
sidebar_state = "collapsed" if st.session_state.app_mode == "COMPARE" else "expanded"

st.set_page_config(
    page_title="Simulation Viewer",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state=sidebar_state
)

SIMULATIONS_DIR = os.path.join(os.path.dirname(__file__), "saved_simulations")

# -----------------------------------------------------------------------------
# 2. Helper Functions (Logic)
# -----------------------------------------------------------------------------

@st.cache_data
def load_simulation_metadata():
    if not os.path.exists(SIMULATIONS_DIR):
        return pd.DataFrame()

    sims = []
    # Scan directory
    for entry in os.scandir(SIMULATIONS_DIR):
        if entry.is_dir():
            sim_data = {
                "Simulation Name": entry.name,
                "path": entry.path,
                "mtime": entry.stat().st_mtime
            }
            
            # Try load config.json
            config_path = os.path.join(entry.path, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        cfg = json.load(f)
                    
                    # Extract Geometry
                    geo = cfg.get("geometry", {})
                    sim_data["Width"] = geo.get("width")
                    sim_data["Length"] = geo.get("length")
                    sim_data["Thickness"] = geo.get("thickness")
                    sim_data["Pivot Depth"] = geo.get("pivot_depth")
                    
                    # Extract Config
                    conf = cfg.get("config", {})
                    sim_data["Timestep (min)"] = conf.get("timestep_min")
                    sim_data["Days"] = conf.get("duration_days")
                    sim_data["Tolerance"] = conf.get("tolerance")
                    sim_data["Plant Rotation"] = conf.get("plant_rotation")
                    sim_data["Grid X"] = conf.get("grid_pitch_x")
                    sim_data["Grid Y"] = conf.get("grid_pitch_y")
                    sim_data["Panels"] = conf.get("total_panels")
                    
                except Exception:
                    pass 
            
            # Check for timeseries file
            ts_path = os.path.join(entry.path, "timeseries.csv")
            sim_data["Has Results"] = os.path.exists(ts_path)
            
            sims.append(sim_data)
    
    if not sims:
        return pd.DataFrame()
        
    df = pd.DataFrame(sims)
    df = df.sort_values("mtime", ascending=False)
    return df

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['Status'] = df['safety'].apply(lambda x: 'CLASH' if x else 'SAFE')
    
    if 'sun_el' in df.columns:
        df['zenith'] = 90 - df['sun_el']
    else:
        df['zenith'] = 0
    
    # Calculate Plot Azimuth (Centered at 0 for South Hemisphere)
    if 'sun_az' in df.columns:
        df['plot_az'] = df['sun_az'].apply(lambda x: x if x <= 180 else x - 360)
        
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df['date'] = df['time'].dt.date
        df['day_of_year'] = df['time'].dt.dayofyear
        df['hour'] = df['time'].dt.hour + df['time'].dt.minute / 60.0
    return df

def get_metadata_html(path):
    sim_dir = os.path.dirname(path)
    config_path = os.path.join(sim_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            geo = cfg.get("geometry", {})
            conf = cfg.get("config", {})
            return f"""
            <div style='font-size: 13px; color: #444; margin-bottom: 5px; background: #fafafa; padding: 5px; border-radius: 4px; border: 1px solid #eee;'>
                <b>Geo:</b> {geo.get('width', '?')}x{geo.get('length', '?')}m • Thk {geo.get('thickness', '?')}m • Piv {geo.get('pivot_depth', '?')}m &nbsp;|&nbsp;
                <b>Grid:</b> {conf.get('grid_pitch_x', '?')}x{conf.get('grid_pitch_y', '?')}m • Rot {conf.get('plant_rotation', '?')}° &nbsp;|&nbsp;
                <b>Sim:</b> {conf.get('total_panels', '?')} pnl • {conf.get('duration_days', '?')} days • Tol {conf.get('tolerance', '?')}m
            </div>
            """
        except: return ""
    return ""

@st.cache_data
def calculate_boundary_shape(data, x_col, y_col, bin_size=1.0):
    if data.empty: return [], []
    d = data[[x_col, y_col]].copy()
    d['bin'] = (d[x_col] / bin_size).round() * bin_size
    d = d.sort_values('bin')
    grouped = d.groupby('bin')[y_col].agg(['min', 'max']).reset_index()
    grouped['gap'] = grouped['bin'].diff() > 5.0
    grouped['group_id'] = grouped['gap'].cumsum()
    shapes = []
    for _, group in grouped.groupby('group_id'):
        x_min = group['bin'].tolist()
        y_min = group['min'].tolist()
        x_max = group['bin'].tolist()
        y_max = group['max'].tolist()
        x_shape = x_min + x_max[::-1] + [x_min[0]]
        y_shape = y_min + y_max[::-1] + [y_min[0]]
        shapes.append((x_shape, y_shape))
    return shapes

def apply_point_sizing(fig, base_size):
    clash_size = base_size
    safe_size = base_size * (2.0/3.0) # 2/3 ratio
    for trace in fig.data:
        if trace.name == "CLASH":
            trace.marker.size = clash_size
        elif trace.name == "SAFE":
            trace.marker.size = safe_size
        else:
            trace.marker.size = safe_size

def process_contour_gaps(lut, gap_threshold=2.0):
    """
    Inserts None (NaN) into LUT if Azimuth gap > threshold.
    Breaks Plotly lines.
    """
    if not lut: return pd.DataFrame(columns=["Azimuth", "Elevation"])
    
    # Sort just in case
    lut = sorted(lut, key=lambda x: x[0])
    
    processed = []
    
    for i, (az, el) in enumerate(lut):
        if i > 0:
            prev_az = lut[i-1][0]
            # Check for gap (handle wrapping if needed, but here simple diff)
            if abs(az - prev_az) > gap_threshold:
                 processed.append([None, None]) # Break
                 
        processed.append([az, el])
        
    return pd.DataFrame(processed, columns=["Azimuth", "Elevation"])

# -----------------------------------------------------------------------------
# 3. Component: Dashboard Renderer
# -----------------------------------------------------------------------------
def render_dashboard(container, path, sim_name, settings):
    """
    Renders the full dashboard into a container (or main st).
    settings: dict with 'show_full_year', 'selected_date', 'point_size'
    """
    with container:
        st.subheader(f"{sim_name}")
        
        # 1. Metadata in Main View
        meta_html = get_metadata_html(path)
        if meta_html:
            st.markdown(meta_html, unsafe_allow_html=True)

        try:
            df = load_data(path)
        except Exception as e:
            st.error(f"Failed to load: {e}")
            return

        # Filtering
        if settings['show_full_year']:
            fg_df = df
            bg_df = pd.DataFrame()
            suffix = "(Full Year)"
        else:
            fg_df = df[df['date'] == settings['selected_date']]
            bg_df = df
            suffix = f"({settings['selected_date']})"
        
        # Stats
        clash_subset = fg_df[fg_df['Status'] == 'CLASH']
        regions = []
        if not clash_subset.empty:
            clash_east = clash_subset[clash_subset['plot_az'] > 0]
            clash_west = clash_subset[clash_subset['plot_az'] <= 0]
            for name, sub in [("East (AM)", clash_east), ("West (PM)", clash_west)]:
                if not sub.empty:
                    regions.append({
                        "name": name,
                        "az_min": sub['plot_az'].min(), "az_max": sub['plot_az'].max(),
                        "el_min": sub['sun_el'].min(), "el_max": sub['sun_el'].max(),
                        "zen_min": sub['zenith'].min(), "zen_max": sub['zenith'].max(),
                        "orig_az_min": sub['sun_az'].min(), "orig_az_max": sub['sun_az'].max()
                    })
            stats_lines = [f"<b>{r['name']}</b>: Az [{r['az_min']:.1f}, {r['az_max']:.1f}] El [{r['el_min']:.1f}, {r['el_max']:.1f}]" for r in regions]
            clash_stats_str = "<br>".join(stats_lines)
        else:
            clash_stats_str = "No Clashes Detected"

        # Helpers for charts
        color_map = {"CLASH": "red", "SAFE": "green"}
        
        def add_layers(fig, x_col, y_col, r_col=None, polar=False):
            # Background
            if not bg_df.empty:
                target_y = r_col if polar else y_col
                # Grey
                bg_shapes = calculate_boundary_shape(bg_df, x_col, target_y)
                for sx, sy in bg_shapes:
                    trace_kwargs = dict(fill='toself', fillcolor='rgba(200, 200, 200, 0.3)', line=dict(color='rgba(200, 200, 200, 0.5)', width=1), name='Full Year Range', hoverinfo='skip')
                    if polar: fig.add_trace(go.Scatterpolar(r=sy, theta=sx, **trace_kwargs))
                    else: fig.add_trace(go.Scatter(x=sx, y=sy, **trace_kwargs))
                # Red
                clash_bg = bg_df[bg_df['Status'] == 'CLASH']
                if not clash_bg.empty:
                    clash_shapes = calculate_boundary_shape(clash_bg, x_col, target_y)
                    for sx, sy in clash_shapes:
                        trace_kwargs = dict(fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', line=dict(color='rgba(255, 0, 0, 0.3)', width=1), name='Known Clash Zones', hoverinfo='skip')
                        if polar: fig.add_trace(go.Scatterpolar(r=sy, theta=sx, **trace_kwargs))
                        else: fig.add_trace(go.Scatter(x=sx, y=sy, **trace_kwargs))
            
            # Foreground
            if polar:
                fg_fig = px.scatter_polar(fg_df, r=r_col, theta=x_col, color="Status", color_discrete_map=color_map, start_angle=0, direction="clockwise", render_mode="webgl", template="plotly_white")
            else:
                fg_fig = px.scatter(fg_df, x=x_col, y=y_col, color="Status", color_discrete_map=color_map, render_mode="webgl", template="plotly_white")
            
            apply_point_sizing(fg_fig, settings['point_size'])
            for trace in fg_fig.data: fig.add_trace(trace)

        # 1. Phase Space (with Stats)
        st.markdown(f"**1. Phase Space** {suffix}")
        fig_phase = go.Figure()
        fig_phase.update_layout(template="plotly_white", margin=dict(t=30, b=10, l=10, r=10))
        add_layers(fig_phase, x_col="plot_az", y_col="sun_el", polar=False)
        for r in regions:
            fig_phase.add_vline(x=r['az_min'], line_width=1, line_dash="solid", line_color="gray", layer="above")
            fig_phase.add_vline(x=r['az_max'], line_width=1, line_dash="solid", line_color="gray", layer="above")
            fig_phase.add_shape(type="rect", x0=r['az_min'], y0=r['el_min'], x1=r['az_max'], y1=r['el_max'], line=dict(color="gray", width=1, dash="solid"), fillcolor="rgba(0,0,0,0)", layer="above")
            fig_phase.add_annotation(x=r['az_min'], y=r['el_max'], text=f"{r['name']}", showarrow=False, yshift=10)
        
        # Overlay Clash Contour (If Exists)
        contour_path = os.path.join(SIMULATIONS_DIR, "Safe Elevation Contours", f"{sim_name}_contour.json")
        if os.path.exists(contour_path):
            try:
                with open(contour_path, 'r') as f:
                    cdata = json.load(f)
                
                # Try new key first, fall back to old
                lut = cdata.get("clash_contour", [])
                line_color = "red"
                line_name = "Clash Contour"
                
                if not lut:
                     lut = cdata.get("safe_elevation_lut", [])
                     if lut: # Old format
                         line_color = "orange"
                         line_name = "Safe Limit (Old)"
                
                if lut:
                    # Convert to Plot Azimuth
                    # [Az, El]
                    # Az is 0-360. 
                    # plot_az logic: x if x <= 180 else x - 360
                    transformed = []
                    for az, el in lut:
                        p_az = az if az <= 180 else az - 360
                        transformed.append([p_az, el])
                    
                    # Process Gaps for Plotting
                    df_seg = process_contour_gaps(transformed, gap_threshold=5.0)
                    
                    fig_phase.add_trace(go.Scatter(
                        x=df_seg['Azimuth'], y=df_seg['Elevation'], 
                        mode='lines', 
                        line=dict(color=line_color, width=2, dash='solid'),
                        name=line_name,
                        connectgaps=False
                    ))
            except Exception:
                pass

        # Add stats to title here
        fig_phase.update_layout(
            xaxis_title="Azimuth (0=N)", 
            yaxis_title="Elevation", 
            xaxis=dict(zeroline=True, zerolinecolor="black"), 
            yaxis=dict(zeroline=True, zerolinecolor="black"),
            title=dict(text=f"<span style='font-size:11px'>{clash_stats_str}</span>", y=0.95)
        )
        st.plotly_chart(fig_phase, use_container_width=True)

        # 2. Clash Calendar
        st.markdown("**2. Clash Calendar**")
        fig_cal = px.scatter(df, x="date", y="hour", color="Status", color_discrete_map=color_map, render_mode="webgl", template="plotly_white")
        apply_point_sizing(fig_cal, settings['point_size'])
        if settings['selected_date']:
            fig_cal.add_vline(x=settings['selected_date'], line_width=2, line_color="orange", line_dash="solid")
        fig_cal.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis_title="Date", yaxis_title="Hour")
        st.plotly_chart(fig_cal, use_container_width=True)

        # 3. Sky Mask (Polar)
        st.markdown(f"**3. Sky Mask (Polar)** {suffix}")
        fig_polar = go.Figure()
        fig_polar.update_layout(template="plotly_white", margin=dict(t=10, b=10, l=10, r=10))
        add_layers(fig_polar, x_col="sun_az", y_col=None, r_col="zenith", polar=True)
        # Boundaries
        for r in regions:
            for az in [r['orig_az_min'], r['orig_az_max']]:
                fig_polar.add_trace(go.Scatterpolar(r=[0, 90], theta=[az, az], mode='lines', line=dict(color='black', width=1, dash='dot'), showlegend=False, hoverinfo='skip'))
            import numpy as np
            theta_range = np.linspace(r['orig_az_min'], r['orig_az_max'], 50)
            for zen in [r['zen_min'], r['zen_max']]:
                fig_polar.add_trace(go.Scatterpolar(r=[zen]*len(theta_range), theta=theta_range, mode='lines', line=dict(color='black', width=1, dash='dot'), showlegend=False, hoverinfo='skip'))
        
        fig_polar.update_layout(polar=dict(radialaxis=dict(range=[0, 90], title="Zenith"), bgcolor="white"))
        st.plotly_chart(fig_polar, use_container_width=True)


# -----------------------------------------------------------------------------
# 4. Main Application Controller
# -----------------------------------------------------------------------------

# Utility Actions
def reset_to_library():
    st.session_state.app_mode = "LIBRARY"
    st.session_state.sim1_path = None
    st.session_state.sim2_path = None
    st.rerun()

def enter_compare_picker():
    st.session_state.app_mode = "PICKING_COMPARE"
    st.rerun()

def exit_compare_to_single():
    st.session_state.app_mode = "SINGLE"
    st.session_state.sim2_path = None
    st.rerun()

# ----------------- VIEW: LIBRARY -----------------
if st.session_state.app_mode == "LIBRARY" or st.session_state.app_mode == "PICKING_COMPARE":
    title = "Simulation Library" if st.session_state.app_mode == "LIBRARY" else "Select Simulation to Compare"
    st.title(title)
    
    if st.session_state.app_mode == "PICKING_COMPARE":
        if st.button("← Cancel Comparison"):
            st.session_state.app_mode = "SINGLE"
            st.rerun()

    df_sims = load_simulation_metadata()
    if not df_sims.empty:
        # Col config
        column_config = {
            "path": None, "mtime": None,
            "Has Results": st.column_config.CheckboxColumn("Results?", disabled=True),
            "Simulation Name": st.column_config.TextColumn("Name", width="medium"),
            "Panels": st.column_config.NumberColumn("Panels", width="small")
        }
        
        selection = st.dataframe(df_sims, use_container_width=True, column_config=column_config, hide_index=True, selection_mode="single-row", on_select="rerun")
        
        if selection and selection.selection.rows:
            idx = selection.selection.rows[0]
            selected_row = df_sims.iloc[idx]
            if selected_row["Has Results"]:
                chosen_path = os.path.join(selected_row["path"], "timeseries.csv")
                chosen_name = selected_row["Simulation Name"]
                
                if st.session_state.app_mode == "LIBRARY":
                    st.session_state.sim1_path = chosen_path
                    st.session_state.sim1_name = chosen_name
                    st.session_state.app_mode = "SINGLE"
                    st.rerun()
                elif st.session_state.app_mode == "PICKING_COMPARE":
                    # Prevent picking same
                    if chosen_path != st.session_state.sim1_path:
                        st.session_state.sim2_path = chosen_path
                        st.session_state.sim2_name = chosen_name
                        st.session_state.app_mode = "COMPARE"
                        st.rerun()
                    else:
                        st.toast("Cannot compare simulation with itself!", icon="⚠️")
    else:
        st.warning("No simulations found.")
        if st.button("Refresh"): st.rerun()

# ----------------- VIEW: SINGLE or COMPARE -----------------
else:
    # --- Sidebar Controls ---
    with st.sidebar:
        if st.session_state.app_mode == "SINGLE":
            st.title("Single View")
            col1, col2 = st.columns(2)
            if col1.button("Library"): reset_to_library()
            if col2.button("Compare"): enter_compare_picker()

            if st.button("Clash Contour Map"): st.session_state.app_mode = "CONTOUR_VIEW"; st.rerun()
        elif st.session_state.app_mode == "CONTOUR_VIEW":
            st.title("Contour Viewer")
            if st.button("← Back to Library"): reset_to_library()
        else:
            st.title("Compare View")
            if st.button("← Stop Comparing"): exit_compare_to_single()
        
        st.markdown("---")
        if st.button("Clear Cache"): st.cache_data.clear(); st.rerun()
        
        st.markdown("### Common Settings")
        show_full = st.toggle("Show Full Year", value=True)
        sel_date = None
        if not show_full:
            # Load date range from Sim1 for slider bounds (assuming similar dates)
            try:
                d1 = pd.read_csv(st.session_state.sim1_path, usecols=['time'])
                d1['time'] = pd.to_datetime(d1['time'])
                min_d, max_d = d1['time'].dt.date.min(), d1['time'].dt.date.max()
                sel_date = st.slider("Select Day", min_d, max_d, min_d)
            except:
                st.warning("Could not read dates for slider")
        
        pt_size = st.slider("Point Size", 0.1, 5.0, 4.0, 0.1)
        
        settings = {
            'show_full_year': show_full,
            'selected_date': sel_date,
            'point_size': pt_size
        }

    # --- Main Canvas ---
    if st.session_state.app_mode == "SINGLE":
        render_dashboard(st.container(), st.session_state.sim1_path, st.session_state.sim1_name, settings)
    
    elif st.session_state.app_mode == "COMPARE":
        c1, c2 = st.columns(2)
        render_dashboard(c1, st.session_state.sim1_path, st.session_state.sim1_name, settings)
        render_dashboard(c2, st.session_state.sim2_path, st.session_state.sim2_name, settings)

    elif st.session_state.app_mode == "CONTOUR_VIEW":
        st.title("Clash Contour Viewer")
        
        CONTOURS_DIR = os.path.join(SIMULATIONS_DIR, "Safe Elevation Contours")
        
        if not os.path.exists(CONTOURS_DIR):
             st.warning(f"No contour maps found in {CONTOURS_DIR}")
        else:
             files = [f for f in os.listdir(CONTOURS_DIR) if f.endswith(".json")]
             if not files:
                 st.warning("No JSON contour maps found.")
             else:
                 selected_file = st.selectbox("Select Contour Map", files)
                 if selected_file:
                     path = os.path.join(CONTOURS_DIR, selected_file)
                     try:
                         with open(path, 'r') as f:
                             data = json.load(f)
                             
                         header = data.get("header", {})
                         
                         # Support both old and new keys for backward compatibility/transition
                         contour_data = data.get("clash_contour", [])
                         if not contour_data:
                              contour_data = data.get("safe_elevation_lut", [])
                         
                         # Metadata
                         geo = header.get("geometry", {})
                         conf = header.get("config", {})
                         sim_source = header.get("source_simulation", "Unknown")
                         gen_at = header.get("generated_at", "Unknown")

                         st.markdown(f"""
                         <div style='font-size: 14px; background: #fafafa; padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-bottom: 20px;'>
                             <b>Source:</b> {sim_source} • <b>Generated:</b> {gen_at}<br>
                             <b>Geometry:</b> {geo.get('width')}x{geo.get('length')}m • Rot {conf.get('plant_rotation')}° • Tol {conf.get('tolerance')}m
                         </div>
                         """, unsafe_allow_html=True)
                         
                         if contour_data:
                             df_lut = process_contour_gaps(contour_data, gap_threshold=5.0)
                             
                             # Plot
                             fig = px.line(df_lut, x="Azimuth", y="Elevation", title="Clash Elevation Contour (Actual Clash Boundary)", markers=True)
                             # connectgaps=False is default in px.line for NaN? Usually yes.
                             fig.update_traces(connectgaps=False, line_shape='linear', line_color='red')
                             
                             fig.update_layout(
                                 xaxis_title="Sun Azimuth (Deg)",
                                 yaxis_title="Min Clash Elevation (Deg)",
                                 template="plotly_white",
                                 yaxis=dict(range=[0, 90])
                             )
                             
                             st.plotly_chart(fig, use_container_width=True)
                             
                             with st.expander("View Raw Contour Data"):
                                 st.dataframe(df_lut, use_container_width=True)
                             
                         else:
                             st.warning("Contour Data is empty.")
                             
                     except Exception as e:
                         st.error(f"Error loading map: {e}")
