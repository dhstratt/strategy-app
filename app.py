import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import pickle
import re

# --- SAFE IMPORT ---
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="The Consumer Landscape")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
        h1, h2, h3 { font-family: 'Nunito', sans-serif; font-weight: 800; }
        .stMetric { font-family: 'Nunito', sans-serif; }
        .mindset-card {
            padding: 20px; border-radius: 10px; border-left: 10px solid #ccc;
            background-color: #f9f9f9; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .size-badge { float: right; background: #004d40; padding: 4px 10px; border-radius: 20px; font-size: 0.85em; font-weight: 800; color: #fff; }
        .size-badge-warning { float: right; background: #e65100; padding: 4px 10px; border-radius: 20px; font-size: 0.85em; font-weight: 800; color: #fff; }
        .code-block { background-color: #1e1e1e; color: #d4d4d4; padding: 25px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 1.1em; margin-top: 10px; white-space: pre-wrap; border: 1px solid #444; line-height: 1.6; }
        .logic-tag { background: #333; color: #fff; padding: 4px 12px; border-radius: 4px; font-size: 0.9em; font-weight: 800; margin-bottom: 15px; display: inline-block; }
        [data-testid="stMetricValue"] { font-size: 1.5rem !important; }
        .status-ok { color: #2e7d32; font-weight: 800; font-size: 1.1em; }
        .status-err { color: #c62828; font-weight: 800; font-size: 1.1em; }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'processed_data' not in st.session_state: st.session_state.processed_data = False
if 'passive_data' not in st.session_state: st.session_state.passive_data = [] 
if 'mindset_report' not in st.session_state: st.session_state.mindset_report = []
if 'universe_size' not in st.session_state: st.session_state.universe_size = 258000.0
if 'df_brands' not in st.session_state: st.session_state.df_brands = pd.DataFrame()
if 'df_attrs' not in st.session_state: st.session_state.df_attrs = pd.DataFrame()
if 'accuracy' not in st.session_state: st.session_state.accuracy = 0

# --- HELPERS ---
def normalize_strings(s_index):
    """Normalize strings for fuzzy matching: lowercase, strip, remove punctuation"""
    return s_index.astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()

def clean_df(df_input, is_core=False):
    """Robust data cleaning: Handles commas, %, $, - and sets index"""
    df = df_input.copy()
    
    # --- AUTO-CLEAN PASSIVE LAYER ---
    label_col = df.columns[0]
    if df[label_col].astype(str).str.contains('_Any Agree', na=False).any():
        df = df[df[label_col].astype(str).str.contains('_Any Agree', na=False)]
        df[label_col] = df[label_col].astype(str).str.replace('_Any Agree', '', regex=False).str.strip()
        df[label_col] = df[label_col].astype(str).str.replace('"', '', regex=False)
    # --------------------------------

    df = df.set_index(label_col)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(r'[,$%]', '', regex=True)
            df[col] = df[col].str.replace(r'^\s*-\s*$', '0', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.fillna(0)
    
    if is_core:
        u_idx = df.index.astype(str).str.contains("Study Universe|Total Population", case=False, regex=True)
        if any(u_idx):
            st.session_state.universe_size = float(df[u_idx].iloc[0].sum())
    
    mask = ~df.index.astype(str).str.contains("Study Universe|Total|Base|Sample", case=False, regex=True)
    valid_cols = [c for c in df.columns if "study universe" not in str(c).lower() and "total" not in str(c).lower()]
    
    return df.loc[mask, valid_cols]

def rotate_coords(df_to_rot, angle_deg):
    theta = np.radians(angle_deg)
    c, s_rot = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s_rot), (s_rot, c)))
    coords = df_to_rot[['x', 'y']].values
    rotated = coords @ R.T
    df_new = df_to_rot.copy()
    df_new['x'], df_new['y'] = rotated[:, 0], rotated[:, 1]
    return df_new

# ==========================================
# DATA PROCESSING ENGINE
# ==========================================
def process_data(uploaded_file, passive_files, passive_configs):
    try:
        uploaded_file.seek(0)
        raw_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df_math_ready = clean_df(raw_data, is_core=True)
        df_math = df_math_ready.loc[(df_math_ready != 0).any(axis=1)]
        
        if not df_math.empty:
            # 1. CORE SVD MAP
            N = df_math.values; matrix_sum = N.sum()
            if matrix_sum == 0: return
            P = N / matrix_sum; r = P.sum(axis=1); c = P.sum(axis=0); E = np.outer(r,c)
            E[E < 1e-9] = 1e-9; R = (P - E) / np.sqrt(E); U, s, Vh = np.linalg.svd(R, full_matrices=False)
            row_coords = (U * s) / np.sqrt(r[:, np.newaxis]); col_coords = (Vh.T * s) / np.sqrt(c[:, np.newaxis])
            
            st.session_state.df_brands = pd.DataFrame(col_coords[:, :2], columns=['x','y']); st.session_state.df_brands['Label'] = df_math.columns
            st.session_state.df_attrs = pd.DataFrame(row_coords[:, :2], columns=['x','y']); st.session_state.df_attrs['Label'] = df_math.index
            st.session_state.df_attrs['Weight'] = df_math.sum(axis=1).values 
            
            eig_vals = np.array(s)**2
            total_var = np.sum(eig_vals)
            st.session_state.accuracy = (np.sum(eig_vals[:2]) / total_var * 100) if total_var > 0 else 0
            st.session_state.processed_data = True

            # 2. PASSIVE LAYER PROJECTION
            pass_list = []
            core_cols_norm = normalize_strings(df_math.columns)
            core_idx_norm = normalize_strings(df_math.index)
            col_mapper = {name: i for i, name in enumerate(core_cols_norm)}
            row_mapper = {name: i for i, name in enumerate(core_idx_norm)}

            for i, cfg in enumerate(passive_configs):
                pf = cfg['file']
                pf.seek(0)
                p_raw = pd.read_csv(pf) if pf.name.endswith('.csv') else pd.read_excel(pf)
                p_c = clean_df(p_raw, is_core=False)
                
                p_cols_norm = normalize_strings(p_c.columns)
                p_idx_norm = normalize_strings(p_c.index)
                
                common_b_count = sum(1 for x in p_cols_norm if x in col_mapper)
                common_r_count = sum(1 for x in p_idx_norm if x in row_mapper)
                is_rows = cfg["mode"] == "Rows (Stars)" if cfg["mode"] != "Auto" else common_b_count > common_r_count
                
                proj = np.array([]); p_aligned = pd.DataFrame()
                status_msg = "❌ No Match"
                
                if is_rows:
                    if common_b_count > 0:
                        status_msg = f"✅ Aligned by {common_b_count} Cols"
                        p_aligned = pd.DataFrame(0.0, index=p_c.index, columns=df_math.columns)
                        for orig_col, norm_col in zip(p_c.columns, p_cols_norm):
                            if norm_col in col_mapper:
                                p_aligned.iloc[:, col_mapper[norm_col]] = p_c[orig_col].values
                        proj = (p_aligned.div(p_aligned.sum(axis=1).replace(0,1), axis=0)).values @ col_coords[:,:2] / s[:2]
                        res = pd.DataFrame(proj, columns=['x','y']); res['Label'] = p_c.index; res['Weight'] = p_c.sum(axis=1).values; res['Shape'] = 'star'
                else:
                    if common_r_count > 0:
                        status_msg = f"✅ Aligned by {common_r_count} Rows"
                        p_aligned = pd.DataFrame(0.0, index=df_math.index, columns=p_c.columns)
                        for orig_row, norm_row in zip(p_c.index, p_idx_norm):
                            if norm_row in row_mapper:
                                p_aligned.iloc[row_mapper[norm_row], :] = p_c.loc[orig_row].values
                        proj = (p_aligned.div(p_aligned.sum(axis=0).replace(0,1), axis=1)).T.values @ row_coords[:,:2] / s[:2]
                        res = pd.DataFrame(proj, columns=['x','y']); res['Label'] = p_c.columns; res['Weight'] = p_c.sum(axis=0).values; res['Shape'] = 'diamond'
                
                if not p_aligned.empty and proj.size > 0:
                    res['LayerName'] = cfg['name']; res['Visible'] = cfg['show']; res['Status'] = status_msg
                    pass_list.append(res)
                else:
                    pass_list.append({'LayerName': cfg['name'], 'Status': "⚠️ Alignment Failed", 'Label': [], 'Visible': False})
            
            st.session_state.passive_data = pass_list
    except Exception as e:
        st.error(f"Processing Error: {e}")

# ==========================================
# UI
# ==========================================
tab1, tab2, tab3 = st.tabs(["🗺️ Strategic Map", "🧹 MRI Cleaner", "📟 Count Code Editor"])

with tab1:
    st.title("🗺️ The Consumer Landscape")
    
    with st.sidebar:
        st.header("📂 Data & Projects")
        with st.expander("💾 Manage Project", expanded=False):
            uploaded_project = st.file_uploader("Load .use file", type=["use"], key="loader")
            if uploaded_project:
                try:
                    p_data = pickle.load(uploaded_project)
                    st.session_state.df_brands = p_data['df_brands']
                    st.session_state.df_attrs = p_data['df_attrs']
                    st.session_state.passive_data = p_data['passive_data']
                    st.session_state.accuracy = p_data['accuracy']
                    st.session_state.universe_size = float(p_data.get('universe_size', 258000.0))
                    st.session_state.processed_data = True
                    st.rerun()
                except: st.error("Load failed.")
            if st.session_state.processed_data:
                proj_name = st.text_input("Project Name", "Strategy_Map")
                proj_dict = {'df_brands': st.session_state.df_brands, 'df_attrs': st.session_state.df_attrs, 'passive_data': st.session_state.passive_data, 'accuracy': st.session_state.accuracy, 'universe_size': st.session_state.universe_size}
                buffer = io.BytesIO(); pickle.dump(proj_dict, buffer); buffer.seek(0)
                st.download_button("Save Project 📥", buffer, f"{proj_name}.use")

        uploaded_file = st.file_uploader("Upload Core Data", type=["csv", "xlsx", "xls"], key="active")
        passive_files = st.file_uploader("Upload Passive Layers", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="passive")
        
        passive_configs = []
        if passive_files:
            st.subheader("⚙️ Layer Manager")
            for i, pf in enumerate(passive_files):
                # Custom Header Logic
                header_name = st.session_state.get(f"n_{pf.name}", pf.name)
                icon = "⚪"
                if st.session_state.processed_data and i < len(st.session_state.passive_data):
                    status = st.session_state.passive_data[i].get('Status', '')
                    if "✅" in status: icon = "🟢"
                    elif "⚠️" in status: icon = "🟡"
                    elif "❌" in status: icon = "🔴"

                with st.expander(f"{icon} {header_name}", expanded=False):
                    p_name = st.text_input("Layer Name", pf.name, key=f"n_{pf.name}")
                    p_show = st.checkbox("Show on Map", True, key=f"s_{pf.name}")
                    p_mode = st.radio("Map As:", ["Auto", "Rows (Stars)", "Columns (Diamonds)"], key=f"mode_{pf.name}")
                    
                    if st.session_state.processed_data and i < len(st.session_state.passive_data):
                        st.caption(f"Status: {st.session_state.passive_data[i].get('Status', 'Pending')}")
                    
                    passive_configs.append({"file": pf, "name": p_name, "show": p_show, "mode": p_mode})

        if uploaded_file: process_data(uploaded_file, passive_files, passive_configs)

        if st.session_state.processed_data:
            st.divider()
            stab = st.session_state.accuracy
            s_col = "#2e7d32" if stab >= 60 else "#c62828"
            st.markdown(f"""
            <div style="background-color: #fff; border: 1px solid #ddd; padding: 10px; border-radius: 5px; border-left: 5px solid {s_col}; text-align: center;">
                <span style="font-size: 0.9em; font-weight: bold; color: #555;">MAP STABILITY</span><br>
                <span style="font-size: 1.6em; font-weight: 800; color: {s_col};">{stab:.1f}%</span><br>
                <span style="font-size: 0.8em; color: {s_col};">{("✅ Stable" if stab>=60 else "⚠️ Unstable")}</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.header("🏷️ Map Controls")
        
        f_brand = "None"
        if st.session_state.processed_data:
            b_list = sorted(st.session_state.df_brands['Label'].tolist(), key=str.casefold)
            f_brand = st.selectbox("Highlight Column:", ["None"] + b_list)

        lbl_cols = st.checkbox("Column Labels", True)
        lbl_rows = st.checkbox("Row Labels", True)
        lbl_passive = st.checkbox("Passive Labels", False)
        
        placeholder_filters = st.container()

        st.divider()
        st.header("⚗️ Mindset Maker")
        enable_clustering = st.checkbox("Enable Mindset Discovery", False) 
        num_clusters = st.slider("Number of Mindsets", 2, 8, 4)
        strictness = st.slider("🎯 Definition Tightness", 0, 100, 30)
        map_rotation = st.slider("🔄 Map Rotation", 0, 360, 0, step=90)

    # --- RENDER ---
    if st.session_state.processed_data:
        df_b = rotate_coords(st.session_state.df_brands.copy(), map_rotation)
        df_a = rotate_coords(st.session_state.df_attrs.copy(), map_rotation)
        p_source = st.session_state.passive_data if 'passive_data' in st.session_state else []
        df_p_list = [rotate_coords(l.copy(), map_rotation) for l in p_source if isinstance(l, pd.DataFrame) and 'x' in l.columns]
        
        mindset_report = []
        df_a['IsCore'] = True
        for l in df_p_list: l['IsCore'] = True

        if enable_clustering and HAS_SKLEARN:
            pool = pd.concat([df_a[['x','y']]] + [l[['x','y']] for l in df_p_list if not l.empty])
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(pool)
            centroids = kmeans.cluster_centers_
            df_a['Cluster'] = kmeans.predict(df_a[['x','y']])
            df_b['Cluster'] = kmeans.predict(df_b[['x','y']])
            for l in df_p_list: l['Cluster'] = kmeans.predict(l[['x','y']])
            
            for i in range(num_clusters):
                c_actives = df_a[df_a['Cluster'] == i].copy()
                c_actives['dist'] = np.sqrt((c_actives['x']-centroids[i][0])**2 + (c_actives['y']-centroids[i][1])**2)
                c_passives = pd.concat([l[l['Cluster'] == i] for l in df_p_list if not l.empty]) if df_p_list else pd.DataFrame()
                if not c_passives.empty: c_passives['dist'] = np.sqrt((c_passives['x']-centroids[i][0])**2 + (c_passives['y']-centroids[i][1])**2)
                
                cluster_sigs = pd.concat([c_actives[['Label','Weight','dist']], c_passives[['Label','Weight','dist']] if not c_passives.empty else None]).dropna()
                if not cluster_sigs.empty:
                    cutoff = np.percentile(cluster_sigs['dist'], 100 - strictness)
                    df_a.loc[df_a['Cluster']==i, 'IsCore'] = c_actives['dist'] <= cutoff
                    for l in df_p_list:
                        m = l['Cluster']==i
                        if any(m): l.loc[m, 'IsCore'] = np.sqrt((l.loc[m,'x']-centroids[i][0])**2 + (l.loc[m,'y']-centroids[i][1])**2) <= cutoff
                    
                    # --- SMART COUNT CODE ALGORITHM (With 40% Cap) ---
                    core_sigs = cluster_sigs[cluster_sigs['dist'] <= cutoff].sort_values('dist').drop_duplicates('Label')
                    raw_pop = core_sigs.head(5)['Weight'].mean()
                    target_pop = min(raw_pop, st.session_state.universe_size * 0.40)
                    pop_pct = (target_pop / st.session_state.universe_size) * 100
                    
                    best_k, best_thresh, best_diff = 10, 6, float('inf')
                    for k in range(5, min(16, len(core_sigs) + 1)):
                        k_items = core_sigs.head(k)
                        sum_weights = k_items['Weight'].sum()
                        min_t = (k // 2) + 1
                        for t in range(min_t, k + 1):
                            est_reach = sum_weights / t
                            diff = abs(est_reach - target_pop)
                            if diff < best_diff:
                                best_diff = diff; best_k = k; best_thresh = t
                    
                    final_items = core_sigs.head(best_k)
                    
                    mindset_report.append({
                        "id": i+1, 
                        "color": px.colors.qualitative.Bold[i % 10], 
                        "rows": final_items['Label'].tolist(), 
                        "pop_000s": target_pop, 
                        "percent": pop_pct, 
                        "brands": df_b[df_b['Cluster']==i]['Label'].tolist(), 
                        "threshold": best_thresh,
                        "is_broad": pop_pct > 40.0
                    })
        else:
            df_a['Cluster'] = 0; df_b['Cluster'] = 0
            for l in df_p_list: l['Cluster'] = 0
        
        # NOTE: If we are not editing, we load the report. 
        # If we ARE editing (session state exists), we use that to override the report display.
        # However, to keep it simple, we initialize the report here, and let the Editor tab Modify it.
        # But for persistent changes to show on Tab 1, we need to check if we have manual overrides.
        
        # Apply Overrides if they exist
        for m in mindset_report:
            ai_hash = hash(tuple(m['rows'])) # Use initial hash as ID
            # Check for manual overrides from Tab 3 using a simpler ID key structure if needed
            # For now, we update the session_state.mindset_report at end of script cycle
            pass

        st.session_state.mindset_report = mindset_report
        
        if enable_clustering:
            v_mode = st.selectbox("👁️ View Focus:", ["Show All"] + [f"Mindset {m['id']}" for m in mindset_report])
        else:
            v_mode = "Show All"
            
        with placeholder_filters:
            with st.expander("🔍 Filter Base Map", expanded=False):
                show_base_cols = st.checkbox("Show Columns (Dots)", True)
                show_base_rows = st.checkbox("Show Rows (Dots)", True)
                sel_brands = st.multiselect("Visible Columns:", sorted(df_b['Label']), default=sorted(df_b['Label']))
                sel_rows = st.multiselect("Visible Rows:", sorted(df_a['Label']), default=sorted(df_a['Label']))
                df_b = df_b[df_b['Label'].isin(sel_brands)]
                df_a = df_a[df_a['Label'].isin(sel_rows)]
            
            for i, layer in enumerate(df_p_list):
                if not layer.empty and layer['Visible'].iloc[0] and 'Label' in layer.columns:
                    with st.expander(f"🔍 Filter {layer['LayerName'].iloc[0]}", expanded=False):
                        l_opts = sorted(layer['Label'].unique())
                        sel_p = st.multiselect("Visible Items:", l_opts, default=l_opts, key=f"filter_{i}")
                        df_p_list[i] = layer[layer['Label'].isin(sel_p)]

        fig = go.Figure()
        
        if enable_clustering:
            for i in range(num_clusters):
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=px.colors.qualitative.Bold[i % 10]), legendgroup=f"M{i+1}", showlegend=True, name=f"Mindset {i+1}"))
        else:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#1f77b4'), name="Columns"))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#d62728'), name="Base Rows"))
            for l in df_p_list:
                if not l.empty:
                    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#555', symbol=l['Shape'].iloc[0]), name=l['LayerName'].iloc[0]))

        def get_so(lbl, base_c, is_core=True):
            if f_brand == "None": return (base_c, 0.9) if is_core else ('#eeeeee', 0.15)
            return (base_c, 1.0) if (lbl == f_brand or lbl in hl) else ('#d3d3d3', 0.2)

        hl = []
        if f_brand != "None":
            hero = df_b[df_b['Label'] == f_brand].iloc[0]
            for d in [df_a] + df_p_list:
                d['temp_d'] = np.sqrt((d['x']-hero['x'])**2 + (d['y']-hero['y'])**2)
                hl += d.sort_values('temp_d').head(5)['Label'].tolist()

        if show_base_cols:
            res = [get_so(r['Label'], '#1f77b4') for _,r in df_b.iterrows()]
            fig.add_trace(go.Scatter(x=df_b['x'], y=df_b['y'], mode='markers', marker=dict(size=12, color=[r[0] for r in res], opacity=[r[1] for r in res], line=dict(width=1, color='white')), text=df_b['Label'], showlegend=False))
            if lbl_cols:
                for _, r in df_b.iterrows():
                    color, opac = get_so(r['Label'], '#1f77b4')
                    fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], showarrow=True, arrowhead=0, arrowcolor=color, ax=0, ay=-20, font=dict(color=color, size=12))

        if show_base_rows:
            if enable_clustering:
                for cid in sorted(df_a['Cluster'].unique()):
                    if v_mode != "Show All" and v_mode != f"Mindset {cid+1}": continue
                    sub = df_a[df_a['Cluster'] == cid]; bc = px.colors.qualitative.Bold[cid % 10]
                    res = [get_so(r['Label'], bc, r.get('IsCore', True)) for _,r in sub.iterrows()]
                    fig.add_trace(go.Scatter(x=sub['x'], y=sub['y'], mode='markers', marker=dict(size=8, color=[r[0] for r in res], opacity=[r[1] for r in res]), text=sub['Label'], legendgroup=f"M{cid+1}", showlegend=False))
                    if lbl_rows:
                        for _, r in sub.iterrows():
                            color, opac = get_so(r['Label'], bc, r.get('IsCore', True))
                            if opac > 0.3: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], showarrow=True, arrowhead=0, arrowcolor=color, ax=0, ay=-15, font=dict(color=color, size=11))
            else:
                res = [get_so(r['Label'], '#d62728') for _,r in df_a.iterrows()]
                fig.add_trace(go.Scatter(x=df_a['x'], y=df_a['y'], mode='markers', marker=dict(size=8, color=[r[0] for r in res], opacity=[r[1] for r in res]), text=df_a['Label'], showlegend=False))
                if lbl_rows:
                    for _, r in df_a.iterrows():
                        color, opac = get_so(r['Label'], '#d62728'); fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], showarrow=True, arrowhead=0, arrowcolor=color, ax=0, ay=-15, font=dict(color=color, size=11))

        for i, layer in enumerate(df_p_list):
            if not layer.empty and layer['Visible'].iloc[0]:
                l_shape = layer['Shape'].iloc[0] 
                if enable_clustering:
                    for cid in sorted(layer['Cluster'].unique()):
                        if v_mode != "Show All" and v_mode != f"Mindset {cid+1}": continue
                        sub = layer[layer['Cluster'] == cid]; bc = px.colors.qualitative.Bold[cid % 10]
                        res = [get_so(r['Label'], bc, r.get('IsCore', True)) for _,r in sub.iterrows()]
                        fig.add_trace(go.Scatter(x=sub['x'], y=sub['y'], mode='markers', marker=dict(size=10, symbol=l_shape, color=[r[0] for r in res], opacity=[r[1] for r in res], line=dict(width=1, color='white')), text=sub['Label'], legendgroup=f"M{cid+1}", showlegend=False))
                        if lbl_passive:
                            for _, r in sub.iterrows():
                                c, o = get_so(r['Label'], bc, r.get('IsCore', True))
                                if o > 0.3: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], showarrow=True, arrowhead=0, arrowcolor=c, ax=0, ay=-15, font=dict(color=c, size=10))
                else:
                    res = [get_so(r['Label'], '#555') for _,r in layer.iterrows()]
                    fig.add_trace(go.Scatter(x=layer['x'], y=layer['y'], mode='markers', marker=dict(size=10, symbol=l_shape, color=[r[0] for r in res], opacity=[r[1] for r in res], line=dict(width=1, color='white')), text=layer['Label'], showlegend=False))
                    if lbl_passive:
                        for _, r in layer.iterrows():
                            color, opac = get_so(r['Label'], '#555'); fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], showarrow=True, arrowhead=0, arrowcolor=color, ax=0, ay=-15, font=dict(color=color, size=10))

        fig.update_layout(template="plotly_white", height=850, yaxis_scaleanchor="x", dragmode='pan')
        st.plotly_chart(fig, use_container_width=True, config={'editable': True, 'scrollZoom': True})

        # --- UPDATED: Use the report that might have been modified by Tab 3 ---
        # We need to access the LATEST state from the Count Code Editor if it exists
        final_report = []
        for i, m in enumerate(st.session_state.mindset_report):
            ai_hash = hash(tuple(m['rows']))
            ms_key = f"ms_items_{m['id']}_{ai_hash}"
            th_key = f"ms_thresh_{m['id']}_{ai_hash}"
            
            # If the user has edited this mindset in Tab 3, we use their values
            if ms_key in st.session_state:
                m['rows'] = st.session_state[ms_key]
            if th_key in st.session_state:
                m['threshold'] = st.session_state[th_key]
                
                # Re-calc size for the Card
                weight_lookup = dict(zip(st.session_state.df_attrs['Label'], st.session_state.df_attrs['Weight']))
                for layer in st.session_state.passive_data:
                    if isinstance(layer, pd.DataFrame) and not layer.empty:
                        weight_lookup.update(dict(zip(layer['Label'], layer['Weight'])))
                
                sum_w = sum([weight_lookup.get(item, 0) for item in m['rows']])
                m['pop_000s'] = sum_w / m['threshold']
                m['percent'] = (m['pop_000s'] / st.session_state.universe_size) * 100
                m['is_broad'] = m['percent'] > 40.0
            
            final_report.append(m)

        if final_report and enable_clustering:
            st.divider(); st.header("👥 Population Analysis")
            cols = st.columns(3)
            for idx, m in enumerate(final_report):
                with cols[idx % 3]:
                    badge_class = "size-badge-warning" if m.get('is_broad', False) else "size-badge"
                    broad_warn = "⚠️ Broad Audience" if m.get('is_broad', False) else ""
                    st.markdown(f"""
                    <div class="mindset-card" style="border-left-color: {m['color']};">
                        <span class="{badge_class}">{m['percent']:.1f}% US</span>
                        <h3 style="color: {m['color']}; margin-top:0;">Mindset {m['id']}</h3>
                        <p style="color: #666; font-size: 0.9em; margin-bottom: 5px;">{broad_warn}</p>
                        <p><b>Vol:</b> {m['pop_000s']:,.0f} (000s)</p>
                        <p><b>Top Signals:</b> {", ".join(m['rows'][:5])}...</p>
                    </div>""", unsafe_allow_html=True)

with tab2:
    st.header("🧹 MRI Data Cleaner")
    raw_mri = st.file_uploader("Upload Raw Export", type=["csv", "xlsx", "xls"])
    if raw_mri:
        try:
            df_r = pd.read_csv(raw_mri, header=None) if raw_mri.name.endswith('.csv') else pd.read_excel(raw_mri, header=None)
            idx = next(i for i, row in df_r.iterrows() if row.astype(str).str.contains("Weighted (000)", regex=False).any())
            brand_r = df_r.iloc[idx-1]; metric_r = df_r.iloc[idx]; data_r = df_r.iloc[idx+1:].copy()
            c_idx, h = [0], ['Attitude']
            for c in range(1, len(metric_r)):
                if "Weighted" in str(metric_r[c]):
                    b_str = str(brand_r[c-1])
                    if "Universe" not in b_str and "Total" not in b_str and b_str != 'nan': c_idx.append(c); h.append(b_str)
            df_c = data_r.iloc[:, c_idx]; df_c.columns = h
            df_c['Attitude'] = df_c['Attitude'].astype(str).str.replace('General Attitudes: ', '', regex=False)
            
            if df_c['Attitude'].astype(str).str.contains('_Any Agree', na=False).any():
                df_c = df_c[df_c['Attitude'].astype(str).str.contains('_Any Agree', na=False)]
                df_c['Attitude'] = df_c['Attitude'].astype(str).str.replace('_Any Agree', '', regex=False).str.strip()
                df_c['Attitude'] = df_c['Attitude'].astype(str).str.replace('"', '', regex=False)
            
            for c in df_c.columns[1:]: df_c[c] = pd.to_numeric(df_c[c].astype(str).str.replace(',', ''), errors='coerce')
            df_c = df_c.dropna(subset=df_c.columns[1:], how='all').fillna(0)
            st.success("Cleaned!"); st.download_button("Download CSV", df_c.to_csv(index=False).encode('utf-8'), "Cleaned_MRI.csv")
        except: st.error("Format error.")

with tab3:
    st.header("📟 Count Code Editor")
    st.markdown("Review the AI's suggested formulas below. You can **manually add or remove traits** and **adjust the threshold** to fine-tune the final audience.")
    
    if not st.session_state.mindset_report: 
        st.warning("Run Discovery on the Strategy Map tab first.")
    else:
        # Build master dictionary for live math
        weight_lookup = dict(zip(st.session_state.df_attrs['Label'], st.session_state.df_attrs['Weight']))
        for layer in st.session_state.passive_data:
            if isinstance(layer, pd.DataFrame) and not layer.empty and 'Label' in layer.columns and 'Weight' in layer.columns:
                weight_lookup.update(dict(zip(layer['Label'], layer['Weight'])))
        
        all_available_labels = sorted(list(weight_lookup.keys()))

        for t in st.session_state.mindset_report:
            with st.expander(f"Mindset {t['id']} Formula Builder", expanded=True):
                ai_hash = hash(tuple(t['rows']))
                ms_key = f"ms_items_{t['id']}_{ai_hash}"
                th_key = f"ms_thresh_{t['id']}_{ai_hash}"
                
                # Initialize state if needed
                if ms_key not in st.session_state: st.session_state[ms_key] = t['rows']
                if th_key not in st.session_state: st.session_state[th_key] = t['threshold']

                st.markdown(f"### <span style='color:{t['color']}'>Mindset {t['id']}</span>", unsafe_allow_html=True)
                
                # 1. Selection
                selected_items = st.multiselect(
                    "Defining Attributes (Add or Remove):", 
                    options=all_available_labels,
                    default=st.session_state[ms_key],
                    key=ms_key
                )
                
                if not selected_items:
                    st.error("Please select at least one attribute.")
                    continue
                    
                # 2. Threshold
                max_thresh = len(selected_items)
                # Ensure threshold is valid for new list size
                if st.session_state[th_key] > max_thresh: st.session_state[th_key] = max_thresh
                    
                selected_thresh = st.slider(
                    "Threshold (Must agree with at least X statements):",
                    min_value=1,
                    max_value=max_thresh,
                    value=st.session_state[th_key],
                    key=th_key
                )
                
                # 3. Live Math
                sum_weights = sum([weight_lookup.get(item, 0) for item in selected_items])
                est_reach_000s = sum_weights / selected_thresh
                est_reach_pct = (est_reach_000s / st.session_state.universe_size) * 100
                
                # 4. Visualization & Warning
                col1, col2 = st.columns(2)
                with col1:
                    # Population Bar Chart
                    bar_color = '#2e7d32' if est_reach_pct <= 40 else '#c62828'
                    st.markdown(f"""
                        <div style="background-color: #eee; border-radius: 5px; width: 100%; height: 20px;">
                            <div style="background-color: {bar_color}; width: {min(100, est_reach_pct)}%; height: 100%; border-radius: 5px;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                            <span style="font-weight: bold; color: {bar_color}">{est_reach_pct:.1f}% Reach</span>
                            <span style="color: #666">40% Cap</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if est_reach_pct > 40:
                        st.caption("⚠️ Audience too broad. Increase threshold.")

                with col2:
                    st.markdown(f"**Population:** {est_reach_000s:,.0f} (000s)")
                    st.markdown(f"*(AI Baseline: ~{t['percent']:.1f}%)*")

                m_code = "(" + " + ".join([f"[{r}]" for r in selected_items]) + f") >= {selected_thresh}"
                st.markdown(f'<div class="logic-tag">MRI SYNTAX</div><div class="code-block">{m_code}</div>', unsafe_allow_html=True)
