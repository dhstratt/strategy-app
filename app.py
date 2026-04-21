import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import textwrap
import math

# --- CONFIGURATION & STYLING ---
st.set_page_config(layout="wide", page_title="Custom CA Studio")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Quicksand', sans-serif; }
        h1, h2, h3 { font-family: 'Quicksand', sans-serif; font-weight: 700; color: #1e1e1e;}
        
        .metric-box { background-color: #f8f9fa; border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 20px;}
        .metric-title { font-size: 0.9em; font-weight: 600; color: #555; text-transform: uppercase;}
        .metric-value { font-size: 1.8em; font-weight: 800; color: #2e7d32; margin: 5px 0;}
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'processed' not in st.session_state: st.session_state.processed = False
if 'df_b_master' not in st.session_state: st.session_state.df_b_master = pd.DataFrame()
if 'df_a_master' not in st.session_state: st.session_state.df_a_master = pd.DataFrame()
if 'passive_layers' not in st.session_state: st.session_state.passive_layers = []
if 'max_dim' not in st.session_state: st.session_state.max_dim = 2
if 's_vals' not in st.session_state: st.session_state.s_vals = []
if 'hidden_items' not in st.session_state: st.session_state.hidden_items = []
if 'map_rot' not in st.session_state: st.session_state.map_rot = 0

# --- CORE MATH FUNCTIONS ---
def normalize_str(s_series):
    return s_series.astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()

def rotate_coords(df, angle_deg):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    coords = df[['x', 'y']].values
    rotated = coords @ R.T
    df_new = df.copy()
    df_new['x'], df_new['y'] = rotated[:, 0], rotated[:, 1]
    return df_new

def process_ca(uploaded_file):
    try:
        # Load and clean basic grid
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
        df = df.set_index(df.columns[0])
        
        # Convert to numeric, forcing errors to NaN then 0
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[,$%]', '', regex=True), errors='coerce')
        df = df.fillna(0)
        
        # Drop "Universe/Total" rows if they exist so they don't anchor the map vertically
        u_idx = df.index.astype(str).str.contains("Study Universe|Total Population|Grand Total", case=False, regex=True)
        df_math = df[~u_idx].copy()
        
        # Drop empty rows
        df_math = df_math.loc[(df_math != 0).any(axis=1)]
        
        N = df_math.values
        matrix_sum = N.sum()
        if matrix_sum == 0: return False
        
        # SVD Math
        P = N / matrix_sum
        r = P.sum(axis=1)
        c = P.sum(axis=0)
        E = np.outer(r, c)
        E[E < 1e-9] = 1e-9
        R = (P - E) / np.sqrt(E)
        U, s, Vh = np.linalg.svd(R, full_matrices=False)
        
        # Save up to 5 dimensions
        st.session_state.max_dim = min(5, len(s))
        st.session_state.s_vals = s
        
        row_coords = (U * s) / np.sqrt(r[:, np.newaxis])
        col_coords = (Vh.T * s) / np.sqrt(c[:, np.newaxis])
        
        # Store Master Data
        st.session_state.df_b_master = pd.DataFrame(col_coords[:, :st.session_state.max_dim], columns=[f'Dim{i+1}' for i in range(st.session_state.max_dim)])
        st.session_state.df_b_master['Label'] = df_math.columns
        
        st.session_state.df_a_master = pd.DataFrame(row_coords[:, :st.session_state.max_dim], columns=[f'Dim{i+1}' for i in range(st.session_state.max_dim)])
        st.session_state.df_a_master['Label'] = df_math.index
        
        st.session_state.processed = True
        return True
    except Exception as e:
        st.error(f"Failed to process file: {e}")
        return False

def process_passive(file, name, mode):
    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
        df = df.set_index(df.columns[0])
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[,$%]', '', regex=True), errors='coerce')
        df = df.fillna(0)
        
        base_cols_norm = normalize_str(st.session_state.df_b_master['Label'])
        base_idx_norm = normalize_str(st.session_state.df_a_master['Label'])
        col_mapper = {n: i for i, n in enumerate(base_cols_norm)}
        row_mapper = {n: i for i, n in enumerate(base_idx_norm)}
        
        max_d = st.session_state.max_dim
        s = st.session_state.s_vals[:max_d]
        
        proj = np.array([])
        shape = 'star'
        
        if mode == "Rows (Match by Columns)":
            p_cols_norm = normalize_str(pd.Series(df.columns))
            if sum(1 for x in p_cols_norm if x in col_mapper) > 0:
                p_aligned = pd.DataFrame(0.0, index=df.index, columns=st.session_state.df_b_master['Label'])
                for orig, norm in zip(df.columns, p_cols_norm):
                    if norm in col_mapper: p_aligned.iloc[:, col_mapper[norm]] = df[orig].values
                # Project
                base_coords = st.session_state.df_b_master[[f'Dim{i+1}' for i in range(max_d)]].values
                proj = (p_aligned.div(p_aligned.sum(axis=1).replace(0,1), axis=0)).values @ base_coords / s
                shape = 'star'
        else:
            p_idx_norm = normalize_str(pd.Series(df.index))
            if sum(1 for x in p_idx_norm if x in row_mapper) > 0:
                p_aligned = pd.DataFrame(0.0, index=st.session_state.df_a_master['Label'], columns=df.columns)
                for orig, norm in zip(df.index, p_idx_norm):
                    if norm in row_mapper: p_aligned.iloc[row_mapper[norm], :] = df.loc[orig].values
                # Project
                base_coords = st.session_state.df_a_master[[f'Dim{i+1}' for i in range(max_d)]].values
                proj = (p_aligned.div(p_aligned.sum(axis=0).replace(0,1), axis=1)).T.values @ base_coords / s
                shape = 'diamond'
                
        if proj.size > 0:
            res = pd.DataFrame(proj, columns=[f'Dim{k+1}' for k in range(max_d)])
            res['Label'] = df.index if mode == "Rows (Match by Columns)" else df.columns
            res['LayerName'] = name
            res['Shape'] = shape
            res['Visible'] = True
            return res
        return None
    except Exception as e:
        st.error(f"Passive Error on {name}: {e}")
        return None

# --- UI LAYOUT ---
st.title("🗺️ CA Presentation Studio")
st.markdown("Upload any raw crosstab grid (Columns = Brands/Groups, Rows = Attributes/Statements). The engine will automatically map the mathematical relationships and prepare them for PowerPoint export.")

# --- SIDEBAR: DATA & MATH ---
with st.sidebar:
    st.header("📂 1. Core Map Data")
    core_file = st.file_uploader("Upload Base Crosstab", type=['csv', 'xlsx'])
    if core_file:
        if st.button("🚀 Run Analysis", use_container_width=True):
            st.session_state.passive_layers = [] # Reset on new core upload
            st.session_state.hidden_items = []
            process_ca(core_file)

    if st.session_state.processed:
        st.header("⚙️ 2. Map Dimensions")
        col_x, col_y = st.columns(2)
        with col_x: x_ax = st.selectbox("X-Axis", range(1, st.session_state.max_dim + 1), index=0)
        with col_y: y_ax = st.selectbox("Y-Axis", range(1, st.session_state.max_dim + 1), index=1 if st.session_state.max_dim > 1 else 0)
        
        st.session_state.map_rot = st.slider("Rotate Map (Degrees)", 0, 360, 0, step=90)
        
        st.header("➕ 3. Passive Layers")
        st.caption("Upload supplementary grids to overlay onto the base map.")
        p_file = st.file_uploader("Upload Passive File", type=['csv', 'xlsx'])
        p_name = st.text_input("Layer Name", "New Layer")
        p_mode = st.radio("Align By:", ["Rows (Match by Columns)", "Columns (Match by Rows)"])
        if p_file and st.button("Overlay Layer"):
            res = process_passive(p_file, p_name, p_mode)
            if res is not None:
                st.session_state.passive_layers.append(res)
                st.success(f"Added {p_name}!")
            else:
                st.error("Could not align layer. Check your column/row names.")
                
        if st.session_state.passive_layers:
            st.markdown("**Active Layers:**")
            for i, layer in enumerate(st.session_state.passive_layers):
                col_tog, col_del = st.columns([4, 1])
                with col_tog:
                    is_vis = st.checkbox(f"👁️ {layer['LayerName'].iloc[0]}", value=layer['Visible'].iloc[0], key=f"vis_{i}")
                    st.session_state.passive_layers[i]['Visible'] = is_vis
                with col_del:
                    if st.button("🗑️", key=f"del_l_{i}", help="Remove Layer"):
                        st.session_state.passive_layers.pop(i)
                        st.rerun()

# --- MAIN CANVAS ---
if st.session_state.processed:
    # Build current view data based on selected axes
    df_b = st.session_state.df_b_master.copy()
    df_b['x'], df_b['y'] = df_b[f'Dim{x_ax}'], df_b[f'Dim{y_ax}']
    
    df_a = st.session_state.df_a_master.copy()
    df_a['x'], df_a['y'] = df_a[f'Dim{x_ax}'], df_a[f'Dim{y_ax}']
    
    df_p_list = []
    for l in st.session_state.passive_layers:
        p_df = l.copy()
        p_df['x'], p_df['y'] = p_df[f'Dim{x_ax}'], p_df[f'Dim{y_ax}']
        df_p_list.append(p_df)

    # Apply Rotation
    if st.session_state.map_rot != 0:
        df_b = rotate_coords(df_b, st.session_state.map_rot)
        df_a = rotate_coords(df_a, st.session_state.map_rot)
        df_p_list = [rotate_coords(p, st.session_state.map_rot) for p in df_p_list]

    # Calculate Stability
    eig = np.array(st.session_state.s_vals)**2
    tot_var = np.sum(eig)
    v_x = (eig[x_ax-1] / tot_var) * 100
    v_y = (eig[y_ax-1] / tot_var) * 100
    stability = v_x + v_y

    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">CURRENT VIEW STABILITY (AXIS {x_ax} + AXIS {y_ax})</div>
            <div class="metric-value" style="color: {'#2e7d32' if stability >= 60 else '#c62828'};">{stability:.1f}%</div>
            <div style="font-size:0.85em; color:#666;">(Axis {x_ax}: {v_x:.1f}% | Axis {y_ax}: {v_y:.1f}%)</div>
        </div>
    """, unsafe_allow_html=True)

    # --- PRESENTATION TOOLBAR ---
    with st.expander("🎨 Visual & Export Settings", expanded=True):
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        with t_col1:
            show_cols = st.checkbox("Show Columns", value=True)
            col_color = st.color_picker("Column Color", "#1f77b4")
            col_shape = st.selectbox("Column Shape", ['circle', 'square', 'diamond', 'star'], index=0)
            col_size = st.slider("Column Dot Size", 5, 30, 16)
        with t_col2:
            show_rows = st.checkbox("Show Rows", value=True)
            row_color = st.color_picker("Row Color", "#d62728")
            row_shape = st.selectbox("Row Shape", ['circle', 'square', 'diamond', 'star'], index=1)
            row_size = st.slider("Row Dot Size", 5, 30, 10)
        with t_col3:
            lbl_pos = st.selectbox("Label Anchor", ["Radial (Auto-Spread)", "Top", "Bottom", "Right", "Left"])
            tail_len = st.slider("Connector Line Length", 10, 100, 30)
            lbl_size = st.slider("Font Size", 8, 24, 12)
        with t_col4:
            # NEW FEATURE: Base Anchor Highlighter
            anchor_col = st.selectbox("Highlight Base/Anchor Column (Plots as ⭐️)", ["None"] + sorted(list(df_b['Label'])))
            wrap_len = st.slider("Max Chars Per Line", 15, 100, 35)
            map_height = st.slider("Canvas Height", 500, 1200, 750, step=50)

    st.button("🔄 Unhide All Labels", on_click=lambda: st.session_state.update({'hidden_items': []}))

    # --- BUILD FIGURE ---
    fig = go.Figure()
    annotations = []

    # Helper for adding traces
    def add_layer_to_fig(df_layer, color, shape, size, name, is_anchor=False):
        if df_layer.empty: return
        
        # Calculate cluster center for radial spread
        cx = float(df_layer['x'].mean())
        cy = float(df_layer['y'].mean())
        
        for _, row in df_layer.iterrows():
            is_visible = row['Label'] not in st.session_state.hidden_items
            wrapped_label = "<br>".join(textwrap.wrap(str(row['Label']), width=wrap_len))
            
            if lbl_pos == "Top": ax, ay = 0, -tail_len
            elif lbl_pos == "Bottom": ax, ay = 0, tail_len
            elif lbl_pos == "Left": ax, ay = -tail_len, 0
            elif lbl_pos == "Right": ax, ay = tail_len, 0
            else: # Radial
                try:
                    dx = float(row['x']) - cx
                    dy = float(row['y']) - cy
                    dist = (dx**2 + dy**2)**0.5
                    
                    if dist > 1e-5: 
                        ax, ay = (dx/dist)*tail_len, -(dy/dist)*tail_len
                    else: 
                        ax, ay = 0, -tail_len
                except Exception:
                    ax, ay = 0, -tail_len

            # Invisible scatter point (for click-to-hide)
            fig.add_trace(go.Scatter(
                x=[row['x']], y=[row['y']], mode='markers',
                marker=dict(size=size, symbol=shape, color=color, line=dict(width=1 if not is_anchor else 2, color='white' if not is_anchor else '#333')),
                customdata=[row['Label']], hovertemplate="<b>%{customdata}</b><extra></extra>",
                name=name, showlegend=False, visible=True if is_visible else False
            ))
            
            # Draggable annotation (Make anchor text bold and larger)
            font_dict = dict(size=lbl_size + (4 if is_anchor else 0), color=color, family="Quicksand")
            annotations.append(dict(
                x=row['x'], y=row['y'], xref="x", yref="y",
                text=f"<b>{wrapped_label}</b>" if is_anchor else wrapped_label, 
                showarrow=True, arrowhead=0, arrowwidth=1 if not is_anchor else 2, arrowcolor=color,
                ax=ax, ay=ay, font=font_dict,
                visible=True if is_visible else False
            ))

    # Add core data, separating out the anchor if one is selected
    if show_cols:
        if anchor_col != "None":
            df_b_normal = df_b[df_b['Label'] != anchor_col]
            df_b_anchor = df_b[df_b['Label'] == anchor_col]
            add_layer_to_fig(df_b_normal, col_color, col_shape, col_size, "Columns")
            # Plot the anchor as a giant black star
            add_layer_to_fig(df_b_anchor, "#111111", "star", col_size + 12, "Base Anchor", is_anchor=True)
        else:
            add_layer_to_fig(df_b, col_color, col_shape, col_size, "Columns")
            
    if show_rows:
        add_layer_to_fig(df_a, row_color, row_shape, row_size, "Rows")
    
    # Add passives
    p_colors = ['#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    for i, p_df in enumerate(df_p_list):
        if p_df.empty or not p_df['Visible'].iloc[0]: continue
        c = p_colors[i % len(p_colors)]
        s = p_df['Shape'].iloc[0]
        n = p_df['LayerName'].iloc[0]
        add_layer_to_fig(p_df, c, s, col_size - 2, n)

    # --- AXIS LOCKING (CRUCIAL FOR PPT OVERLAYS) ---
    all_x = df_b['x'].tolist() + df_a['x'].tolist()
    all_y = df_b['y'].tolist() + df_a['y'].tolist()
    for p in df_p_list:
        all_x.extend(p['x'].tolist()); all_y.extend(p['y'].tolist())
    
    max_val = max(np.max(np.abs(all_x)), np.max(np.abs(all_y))) * 1.15 if all_x else 1

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', dragmode='pan',
        margin=dict(l=0, r=0, t=0, b=0), height=map_height,
        xaxis=dict(range=[-max_val, max_val], fixedrange=False, zeroline=True, zerolinecolor='#eee', showgrid=False, showticklabels=False),
        yaxis=dict(range=[-max_val, max_val], fixedrange=False, zeroline=True, zerolinecolor='#eee', showgrid=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        annotations=annotations
    )

    exp_config = {
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
        'scrollZoom': True,
        'edits': {'annotationTail': True, 'annotationText': True, 'annotationPosition': False},
        'toImageButtonOptions': {'format': 'png', 'filename': "CA_Export", 'height': 720, 'width': 1280, 'scale': 3}
    }

    st.info("💡 **Instructions:** You can now zoom and pan around the map! Click and drag any text label to un-clutter the map. Click directly on a dot to hide it entirely. Hover over the top right to download a high-res 16:9 PNG for PowerPoint. (Tip: Use the 'Reset Axes' house icon before exporting so your images stack perfectly in PPT!)")

    # Render Chart
    map_event = st.plotly_chart(
        fig, use_container_width=True, config=exp_config,
        on_select="rerun", selection_mode="points", key="main_studio_map"
    )

    # Click-to-Hide Logic
    if map_event and map_event.selection.get("points"):
        clicked_pts = [pt["customdata"] for pt in map_event.selection["points"] if "customdata" in pt]
        if clicked_pts:
            changed = False
            for cp in clicked_pts:
                if cp not in st.session_state.hidden_items:
                    st.session_state.hidden_items.append(cp)
                    changed = True
            if changed: st.rerun()

else:
    st.info("👈 Upload a Core Data crosstab in the sidebar to begin building your map.")
