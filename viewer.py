import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Simulation Viewer",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. File Management & Sidebar
# -----------------------------------------------------------------------------
SIMULATIONS_DIR = os.path.join(os.path.dirname(__file__), "saved_simulations")

st.sidebar.title("Simulation Viewer")

# handle cache clearing
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.rerun()

# Check direction exists
if not os.path.exists(SIMULATIONS_DIR):
    st.error(f"Directory not found: `{SIMULATIONS_DIR}`")
    st.sidebar.warning("Please ensure queries are saved in 'saved_simulations/'.")
    st.stop()

# Scan for simulation directories (folders containing timeseries.csv)
# Structure: saved_simulations/<sim_name>/timeseries.csv
sim_folders = [f.path for f in os.scandir(SIMULATIONS_DIR) if f.is_dir()]

valid_sims = []
for folder in sim_folders:
    ts_path = os.path.join(folder, "timeseries.csv")
    if os.path.exists(ts_path):
        valid_sims.append({
            "name": os.path.basename(folder),
            "path": ts_path,
            "mtime": os.path.getmtime(ts_path)
        })

if not valid_sims:
    st.warning(f"No simulation folders with `timeseries.csv` found in `{SIMULATIONS_DIR}`.")
    st.stop()

# Sort by modification time (newest first)
valid_sims.sort(key=lambda x: x["mtime"], reverse=True)

sim_names = [s["name"] for s in valid_sims]

selected_sim_name = st.sidebar.selectbox(
    "Select Simulation Result",
    options=sim_names,
    index=0
)

# Find selected path
selected_sim = next(s for s in valid_sims if s["name"] == selected_sim_name)
file_path = selected_sim["path"]

# -----------------------------------------------------------------------------
# 3. Data Loading & Pre-processing
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(path):
    # Load CSV
    df = pd.read_csv(path)
    
    # Pre-processing
    df['Status'] = df['safety'].apply(lambda x: 'CLASH' if x else 'SAFE')
    
    # Zenith Calculation: 90 - sun_el
    if 'sun_el' in df.columns:
        df['zenith'] = 90 - df['sun_el']
    else:
        df['zenith'] = 0
    
    # Time Parsing
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df['date'] = df['time'].dt.date
        df['day_of_year'] = df['time'].dt.dayofyear
        df['hour'] = df['time'].dt.hour + df['time'].dt.minute / 60.0
        
    return df

with st.spinner("Loading simulation data..."):
    # Load data
    try:
        df = load_data(file_path)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

# Metrics sidebar
st.sidebar.markdown("---")
st.sidebar.info(f"**Rows loaded:** {len(df):,}")
clash_count = len(df[df['Status'] == 'CLASH'])
st.sidebar.metric("Clash Points", f"{clash_count:,}")

# -----------------------------------------------------------------------------
# 4. View Settings & Filtering
# -----------------------------------------------------------------------------
st.sidebar.markdown("### View Settings")
show_full_year = st.sidebar.toggle("Show Full Year", value=True)

selected_date = None
if not show_full_year:
    min_date = df['date'].min()
    max_date = df['date'].max()
    selected_date = st.sidebar.slider(
        "Select Day",
        min_value=min_date,
        max_value=max_date,
        value=min_date
    )

# Prepare Data Subsets
# If Full Year: foreground = all data, background = empty
# If Single Day: foreground = day data, background = all data (dimmed)

if show_full_year:
    fg_df = df
    bg_df = pd.DataFrame() # Empty
    title_suffix = "(Full Year)"
else:
    fg_df = df[df['date'] == selected_date]
    bg_df = df
    title_suffix = f"({selected_date})"

# -----------------------------------------------------------------------------
# 5. Visualizations
# -----------------------------------------------------------------------------

st.title(f"Analysis: {selected_sim_name}")

# Shared color map
color_map = {
    "CLASH": "red",
    "SAFE": "green"
}

# --- Shared Helper for Layering ---
@st.cache_data
def calculate_boundary_shape(data, x_col, y_col, bin_size=1.0):
    """
    Calculates a polygon envelope (min/max y for each x bin) for the data.
    Returns x_coords, y_coords list for a closed shape.
    Splits data into clusters if there are large gaps in X to avoid connecting disjoint regions.
    """
    if data.empty:
        return [], []
    
    # Work with a copy and round X to bin_size
    d = data[[x_col, y_col]].copy()
    d['bin'] = (d[x_col] / bin_size).round() * bin_size
    
    # Sort by bin
    d = d.sort_values('bin')
    
    # Group by bin to get min/max
    grouped = d.groupby('bin')[y_col].agg(['min', 'max']).reset_index()
    
    # Detect gaps to split shapes (e.g. East vs West clashes)
    # If gap > 5 degrees, split
    grouped['gap'] = grouped['bin'].diff() > 5.0
    grouped['group_id'] = grouped['gap'].cumsum()
    
    shapes = []
    
    for _, group in grouped.groupby('group_id'):
        # Construct the polygon path: Min line -> Max line reversed
        x_min = group['bin'].tolist()
        y_min = group['min'].tolist()
        
        x_max = group['bin'].tolist()
        y_max = group['max'].tolist()
        
        # Closed loop
        x_shape = x_min + x_max[::-1] + [x_min[0]]
        y_shape = y_min + y_max[::-1] + [y_min[0]]
        
        shapes.append((x_shape, y_shape))
        
    return shapes

def add_layers(fig, x_col, y_col, r_col=None, polar=False):
    # If background data exists (Single Day mode), draw SHAPES for perfromance
    if not bg_df.empty:
        # 1. Full Year Background (Grey Area) - Represents all valid sun positions
        # Using All Data for the track
        # Assuming bg_df is the full dataset
        
        # Calculate shapes
        # For Polar: X=theta(az), Y=r(zenith)
        # For Cartesian: X=az, Y=el
        
        target_y = r_col if polar else y_col
        
        # Shape 1: The "Safe" or "Full Area" Band (All points)
        # Optimization: We just calculate the hull of everything, color it grey
        bg_shapes = calculate_boundary_shape(bg_df, x_col, target_y)
        
        for sx, sy in bg_shapes:
            if polar:
                fig.add_trace(go.Scatterpolar(
                    r=sy, theta=sx,
                    fill='toself', fillcolor='rgba(200, 200, 200, 0.3)',
                    line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
                    name='Full Year Range', hoverinfo='skip'
                ))
            else:
                 fig.add_trace(go.Scatter(
                    x=sx, y=sy,
                    fill='toself', fillcolor='rgba(200, 200, 200, 0.3)',
                    line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
                    name='Full Year Range', hoverinfo='skip'
                ))

        # Shape 2: The "Clash" Band (All Clashes) -> Red Area
        clash_bg = bg_df[bg_df['Status'] == 'CLASH']
        if not clash_bg.empty:
             clash_shapes = calculate_boundary_shape(clash_bg, x_col, target_y)
             for sx, sy in clash_shapes:
                if polar:
                    fig.add_trace(go.Scatterpolar(
                        r=sy, theta=sx,
                        fill='toself', fillcolor='rgba(255, 0, 0, 0.2)',
                        line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                        name='Known Clash Zones', hoverinfo='skip'
                    ))
                else:
                     fig.add_trace(go.Scatter(
                        x=sx, y=sy,
                        fill='toself', fillcolor='rgba(255, 0, 0, 0.2)',
                        line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                        name='Known Clash Zones', hoverinfo='skip'
                    ))
    
    # Add Foreground Data (Actual colors)
    if polar:
        fg_fig = px.scatter_polar(
            fg_df, r=r_col, theta=x_col, color="Status",
            color_discrete_map=color_map,
            start_angle=0, direction="clockwise",
            render_mode="webgl", hover_data=['sun_el', 'sun_az', 'time', 'act_w', 'theo_w']
        )
    else:
        fg_fig = px.scatter(
            fg_df, x=x_col, y=y_col, color="Status",
            color_discrete_map=color_map,
            render_mode="webgl", hover_data=['time']
        )
        
    for trace in fg_fig.data:
        fig.add_trace(trace)

# Filter clashes for stats (Global or Local? Usually Stats for the VIEWED data is best)
clash_subset = fg_df[fg_df['Status'] == 'CLASH']
regions = []

if not clash_subset.empty:
    clash_east = clash_subset[clash_subset['sun_az'] < 180]
    clash_west = clash_subset[clash_subset['sun_az'] >= 180]
    
    if not clash_east.empty:
        regions.append({
            "name": "East (AM)",
            "az_min": clash_east['sun_az'].min(), "az_max": clash_east['sun_az'].max(),
            "el_min": clash_east['sun_el'].min(), "el_max": clash_east['sun_el'].max(),
            "zen_min": clash_east['zenith'].min(), "zen_max": clash_east['zenith'].max()
        })
    if not clash_west.empty:
        regions.append({
            "name": "West (PM)",
            "az_min": clash_west['sun_az'].min(), "az_max": clash_west['sun_az'].max(),
            "el_min": clash_west['sun_el'].min(), "el_max": clash_west['sun_el'].max(),
            "zen_min": clash_west['zenith'].min(), "zen_max": clash_west['zenith'].max()
        })
    
    stats_lines = []
    for r in regions:
        stats_lines.append(f"<b>{r['name']}</b>: Az [{r['az_min']:.1f}, {r['az_max']:.1f}] El [{r['el_min']:.1f}, {r['el_max']:.1f}]")
    clash_stats_str = "<br>".join(stats_lines)
else:
    clash_stats_str = "No Clashes Detected"


# --- Chart 1: The Sky Mask (Polar Plot) ---
st.subheader(f"1. The Sky Mask (Polar Plot) {title_suffix}")
st.markdown("Relative to sky dome. **Center**: Zenith (0°). **Edge**: Horizon.")

fig_polar = go.Figure()
add_layers(fig_polar, x_col="sun_az", y_col=None, r_col="zenith", polar=True)

# Add boundary lines
for r in regions:
    for az in [r['az_min'], r['az_max']]:
        fig_polar.add_trace(go.Scatterpolar(
            r=[0, 90], theta=[az, az],
            mode='lines', line=dict(color='white', width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))
    import numpy as np
    theta_range = np.linspace(r['az_min'], r['az_max'], 50)
    for zen in [r['zen_min'], r['zen_max']]:
        fig_polar.add_trace(go.Scatterpolar(
            r=[zen]*len(theta_range), theta=theta_range,
            mode='lines', line=dict(color='white', width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))

fig_polar.update_layout(
    polar=dict(radialaxis=dict(range=[0, 90], title="Zenith Angle")),
    title=f"Clash Atlas<br><span style='font-size:12px'>{clash_stats_str}</span>",
    legend_title="Status"
)
st.plotly_chart(fig_polar, use_container_width=True)


# --- Chart 2: The Phase Space (Cartesian Scatter) ---
st.subheader(f"2. The Phase Space (X/Y Map) {title_suffix}")
st.markdown("Linear map for fitting safety boundaries.")

fig_phase = go.Figure()
add_layers(fig_phase, x_col="sun_az", y_col="sun_el", polar=False)

# Add shapes
for r in regions:
    fig_phase.add_vline(x=r['az_min'], line_width=1, line_dash="dash", line_color="white")
    fig_phase.add_vline(x=r['az_max'], line_width=1, line_dash="dash", line_color="white")
    fig_phase.add_shape(type="rect",
        x0=r['az_min'], y0=r['el_min'], x1=r['az_max'], y1=r['el_max'],
        line=dict(color="white", width=1, dash="dash"),
        fillcolor="rgba(0,0,0,0)"
    )
    fig_phase.add_annotation(x=r['az_min'], y=r['el_max'], text=f"{r['name']}", showarrow=False, yshift=10)

fig_phase.update_layout(
    title=f"Sun Elevation vs Azimuth<br><span style='font-size:12px'>{clash_stats_str}</span>",
    xaxis_title="Sun Azimuth (deg)",
    yaxis_title="Sun Elevation (deg)"
)
st.plotly_chart(fig_phase, use_container_width=True)


# --- New Chart: Clash Calendar ---
st.subheader("3. Clash Calendar (Seasonal Analysis)")
st.markdown("Identify time-of-day and seasonal patterns of clashes.")

# For Calendar, we want Full Year data always visible, but maybe emphasize selection?
# Actually, Calendar is best for "Full Year" view.
# Heatmap or Scatter? Scatter allows seeing individual points.
# X: Date, Y: Hour
# Color: Status

fig_calendar = px.scatter(
    df, # Provide full dataframe to see context
    x="date",
    y="hour",
    color="Status",
    color_discrete_map=color_map,
    render_mode="webgl",
    title="Clash Distribution: Date vs Hour",
    labels={"date": "Date", "hour": "Hour of Day (Decimal)"},
    hover_data=['sun_az', 'sun_el']
)
# If a day is selected, add a vertical line marker
if selected_date:
    fig_calendar.add_vline(x=selected_date, line_width=2, line_color="yellow", line_dash="solid")

st.plotly_chart(fig_calendar, use_container_width=True)


# --- Chart 4: Yield Impact Timeline ---
st.subheader(f"4. Yield Impact Timeline {title_suffix}")
st.markdown("Energy production over time.")

# For Yield, if single day selected, zoom to it.
subset_for_yield = fg_df # Rename for clarity

trace_theo = go.Scatter(
    x=subset_for_yield['time'],
    y=subset_for_yield['theo_w'],
    mode='lines',
    name='Theoretical Yield',
    line=dict(color='grey', dash='dash', width=1),
    opacity=0.7
)

trace_act = go.Scatter(
    x=subset_for_yield['time'],
    y=subset_for_yield['act_w'],
    mode='lines',
    fill='tozeroy',
    name='Actual Yield',
    line=dict(color='green', width=1)
)

clashes_yield = subset_for_yield[subset_for_yield['Status'] == 'CLASH']
trace_clash = go.Scatter(
    x=clashes_yield['time'],
    y=[0] * len(clashes_yield),
    mode='markers',
    name='Clash Event',
    marker=dict(color='red', symbol='x', size=8)
)

fig_timeline = go.Figure(data=[trace_theo, trace_act, trace_clash])
fig_timeline.update_layout(
    title="Yield vs Time",
    xaxis_title="Time",
    yaxis_title="Power (W)",
    hovermode="x unified"
)
st.plotly_chart(fig_timeline, use_container_width=True)
