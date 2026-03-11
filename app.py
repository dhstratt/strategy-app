import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import pickle
import re
import math 

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
        .custom-mindset-card {
            padding: 20px; border-radius: 10px; border-left: 10px solid #673ab7;
            background-color: #f3e5f5; margin-bottom: 25px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .size-badge { float: right; background: #004d40; padding: 4px 10px; border-radius: 20px; font-size: 0.85em; font-weight: 800; color: #fff; }
        .size-badge-warning { float: right; background: #e65100; padding: 4px 10px; border-radius: 20px; font-size: 0.85em; font-weight: 800; color: #fff; }
        .size-badge-custom { float: right; background: #4527a0; padding: 4px 10px; border-radius: 20px; font-size: 0.85em; font-weight: 800; color: #fff; }
        .code-block { background-color: #1e1e1e; color: #d4d4d4; padding: 25px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 1.1em; margin-top: 10px; white-space: pre-wrap; border: 1px solid #444; line-height: 1.6; }
        .logic-tag { background: #333; color: #fff; padding: 4px 12px; border-radius: 4px; font-size: 0.9em; font-weight: 800; margin-bottom: 15px; display: inline-block; }
        .success-box { background-color: #e8f5e9; border: 1px solid #4caf50; padding: 10px; border-radius: 5px; color: #2e7d32; font-weight: bold; margin-top: 10px; height: 100%; display: flex; align-items: center;}
        .calibration-box { background-color: #e8f5e9; border: 1px solid #4caf50; padding: 10px; border-radius: 5px; color: #2e7d32; font-weight: bold; margin-bottom: 15px;}
        .error-box { background-color: #ffebee; border: 1px solid #f44336; padding: 15px; border-radius: 5px; color: #c62828; margin-bottom: 15px; font-weight: 600;}
        [data-testid="stMetricValue"] { font-size: 1.5rem !important; }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'processed_data' not in st.session_state: st.session_state.processed_data = False
if 'passive_data' not in st.session_state: st.session_state.passive_data = [] 
if 'universe_size' not in st.session_state: st.session_state.universe_size = 258000.0
if 'exact_universe_found' not in st.session_state: st.session_state.exact_universe_found = False
if 'df_brands' not in st.session_state: st.session_state.df_brands = pd.DataFrame()
if 'df_attrs' not in st.session_state: st.session_state.df_attrs = pd.DataFrame()
if 'accuracy' not in st.session_state: st.session_state.accuracy = 0
if 'lasso_labels' not in st.session_state: st.session_state.lasso_labels = []
if 'map_key' not in st.session_state: st.session_state.map_key = 0 
if 'mindset_report' not in st.session_state: st.session_state.mindset_report = []
if 'corr_matrix' not in st.session_state: st.session_state.corr_matrix = pd.DataFrame()
if 'saved_mindsets' not in st.session_state: st.session_state.saved_mindsets = []

# --- HELPERS ---
def normalize_strings(s_index):
    return s_index.astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()

def truncate_label(text, max_words):
    words = str(text).split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return text

def clean_df(df_input, is_core=False):
    df = df_input.copy()
    label_col = df.columns[0]
    
    df[label_col] = df[label_col].astype(str)
    df[label_col] = df[label_col].str.replace('General Attitudes: ', '', case=False, regex=True)
    df[label_col] = df[label_col].str.replace('_Any Agree', '', regex=False)
    df[label_col] = df[label_col].str.replace('"', '', regex=False)
    df[label_col] = df[label_col].str.strip()
    
    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    df = df.set_index(label_col)
    
    # Drop duplicate rows (safeguard against messy MRI files)
    df = df[~df.index.duplicated(keep='first')]
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(r'[,$%]', '', regex=True)
            df[col] = df[col].str.replace(r'^\s*-\s*$', '0', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    
    if is_core:
        u_idx = df.index.astype(str).str.contains("Study Universe|Total Population", case=False, regex=True)
        if any(u_idx):
            calc_size = float(df[u_idx].iloc[0].sum())
            if calc_size > 0:
                st.session_state.universe_size = calc_size
                st.session_state.exact_universe_found = True
            else:
                st.session_state.exact_universe_found = False
        else:
            st.session_state.exact_universe_found = False
            st.session_state.universe_size = 258000.0
    
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

def process_correlation_matrix(uploaded_file):
    try:
        # Securely read Bytes to prevent buffer reset
        file_bytes = uploaded_file.getvalue()
        df_corr = pd.read_csv(io.BytesIO(file_bytes), index_col=0) if uploaded_file.name.endswith('.csv') else pd.read_excel(io.BytesIO(file_bytes), index_col=0)
        df_corr.index = normalize_strings(df_corr.index)
        df_corr.columns = normalize_strings(df_corr.columns)
        for col in df_corr.columns:
            df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce')
        df_corr = df_corr.fillna(0)
        st.session_state.corr_matrix = df_corr
        return True
    except Exception as e:
        st.error(f"Error parsing correlation matrix: {e}")
        return False

def calculate_clustered_reach(weights_list, threshold, universe_size, overlap_factor):
    if not weights_list or threshold == 0: return 0
    
    N = len(weights_list)
    K = threshold
    perfect_reach = np.mean(weights_list)
    
    avg_p = np.mean(weights_list) / universe_size
    if avg_p >= 1.0: avg_p = 0.99
    if avg_p <= 0.0: return perfect_reach 
    
    indep_prob = 0
    for k in range(K, N + 1):
        if avg_p == 0:
            term = 0 if k > 0 else 1
        elif avg_p == 1:
            term = 1 if k == N else 0
        else:
            term = math.comb(N, k) * (avg_p**k) * ((1 - avg_p)**(N - k))
        indep_prob += term
    
    indep_reach = indep_prob * universe_size
    estimated_reach = (overlap_factor * perfect_reach) + ((1.0 - overlap_factor) * indep_reach)
    return estimated_reach

def get_safe_universe():
    return st.session_state.universe_size if st.session_state.universe_size > 0 else 258000.0

# ==========================================
# DATA PROCESSING ENGINE
# ==========================================
def process_data(uploaded_file, passive_files, passive_configs):
    try:
        # Secure core data buffer
        core_bytes = uploaded_file.getvalue()
        raw_data = pd.read_csv(io.BytesIO(core_bytes)) if uploaded_file.name.endswith('.csv') else pd.read_excel(io.BytesIO(core_bytes))
        df_math_ready = clean_df(raw_data, is_core=True)
        df_math = df_math_ready.loc[(df_math_ready != 0).any(axis=1)]
        
        if not df_math.empty:
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

            pass_list = []
            core_cols_norm = normalize_strings(df_math.columns)
            core_idx_norm = normalize_strings(df_math.index)
            col_mapper = {name: i for i, name in enumerate(core_cols_norm)}
            row_mapper = {name: i for i, name in enumerate(core_idx_norm)}

            for i, cfg in enumerate(passive_configs):
                pf = cfg['file']
                # Secure passive buffer
                pf_bytes = pf.getvalue()
                p_raw = pd.read_csv(io.BytesIO(pf_bytes)) if pf.name.endswith('.csv') else pd.read_excel(io.BytesIO(pf_bytes))
                p_c = clean_df(p_raw, is_core=False)
                
                p_cols_norm = normalize_strings(p_c.columns)
                p_idx_norm = normalize_strings(p_c.index)
                
                common_b_count = sum(1 for x in p_cols_norm if x in col_mapper)
                common_r_count = sum(1 for x in p_idx_norm if x in row_mapper)
                
                # --- SAFER RADIO LOGIC ---
                is_rows = True
                if cfg["mode"] == "Rows (Stars)":
                    is_rows = True
                elif cfg["mode"] == "Columns (Diamonds)":
                    is_rows = False
                else:
                    is_rows = common_b_count >= common_r_count
                
                proj = np.array([]); p_aligned = pd.DataFrame()
                status_msg = "❌ No Match"
                
                if is_rows:
                    if common_b_count > 0:
                        status_msg = f"✅ Aligned by {common_b_count} Columns"
                        p_aligned = pd.DataFrame(0.0, index=p_c.index, columns=df_math.columns)
                        for orig_col, norm_col in zip(p_c.columns, p_cols_norm):
                            if norm_col in col_mapper:
                                p_aligned.iloc[:, col_mapper[norm_col]] = p_c[orig_col].values
                        proj = (p_aligned.div(p_aligned.sum(axis=1).replace(0,1), axis=0)).values @ col_coords[:,:2] / s[:2]
                        res = pd.DataFrame(proj, columns=['x','y']); res['Label'] = p_c.index; res['Weight'] = p_c.sum(axis=1).values; res['Shape'] = 'star'
                    else:
                        status_msg = "❌ Force 'Rows' failed: No matching columns found."
                else:
                    if common_r_count > 0:
                        status_msg = f"✅ Aligned by {common_r_count} Rows"
                        p_aligned = pd.DataFrame(0.0, index=df_math.index, columns=p_c.columns)
                        for orig_row, norm_row in zip(p_c.index, p_idx_norm):
                            if norm_row in row_mapper:
                                p_aligned.iloc[row_mapper[norm_row], :] = p_c.loc[orig_row].values
                        proj = (p_aligned.div(p_aligned.sum(axis=0).replace(0,1), axis=1)).T.values @ row_coords[:,:2] / s[:2]
                        res = pd.DataFrame(proj, columns=['x','y']); res['Label'] = p_c.columns; res['Weight'] = p_c.sum(axis=0).values; res['Shape'] = 'diamond'
                    else:
                        status_msg = "❌ Force 'Columns' failed: No matching rows found."
                
                if not p_aligned.empty and proj.size > 0:
                    res['LayerName'] = cfg['name']; res['Visible'] = cfg['show']; res['Status'] = status_msg
                    pass_list.append(res)
                else:
                    pass_list.append({'LayerName': cfg['name'], 'Status': status_msg, 'Label': [], 'Visible': False})
            
            st.session_state.passive_data = pass_list
    except Exception as e:
        st.error(f"Processing Error: {e}")

# ==========================================
# UI
# ==========================================
tab1, tab2, tab3 = st.tabs(["🗺️ Strategic Map", "🧹 MRI Cleaner", "📟 Count Code Editor"])

with tab1:
    st.title("🗺️ The Consumer Landscape")
    
    if st.session_state.processed_data and not st.session_state.exact_universe_found:
        st.warning("⚠️ **Warning:** A 'Study Universe' or 'Total Population' row was not found in your base data. The app is using a default estimate of 258,000 (000s) to calculate reach percentages.")
        
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
                    st.session_state.exact_universe_found = p_data.get('exact_universe_found', False)
                    st.session_state.corr_matrix = p_data.get('corr_matrix', pd.DataFrame())
                    st.session_state.saved_mindsets = p_data.get('saved_mindsets', []) 
                    st.session_state.processed_data = True
                    st.rerun()
                except: st.error("Load failed.")
            if st.session_state.processed_data:
                proj_name = st.text_input("Project Name", "Strategy_Map")
                proj_dict = {
                    'df_brands': st.session_state.df_brands, 
                    'df_attrs': st.session_state.df_attrs, 
                    'passive_data': st.session_state.passive_data, 
                    'accuracy': st.session_state.accuracy, 
                    'universe_size': st.session_state.universe_size,
                    'exact_universe_found': st.session_state.exact_universe_found,
                    'corr_matrix': st.session_state.corr_matrix,
                    'saved_mindsets': st.session_state.saved_mindsets 
                }
                buffer = io.BytesIO(); pickle.dump(proj_dict, buffer); buffer.seek(0)
                st.download_button("Save Project 📥", buffer, f"{proj_name}.use")

        uploaded_file = st.file_uploader("Upload Core Data", type=["csv", "xlsx", "xls"], key="active")
        passive_files = st.file_uploader("Upload Passive Layers", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="passive")
        
        passive_configs = []
        if passive_files:
            st.subheader("⚙️ Layer Manager")
            for i, pf in enumerate(passive_files):
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

        st.divider()
        st.header("🔗 Advanced Calibration")
        st.markdown("<span style='font-size: 0.85em; color: #555;'>Upload an MRI Cross-Tab/Correlation Matrix to activate strictly accurate population math in the Count Code Editor.</span>", unsafe_allow_html=True)
        corr_file = st.file_uploader("Upload Matrix File", type=["csv", "xlsx", "xls"], key="corr_upload")
        if corr_file:
            if process_correlation_matrix(corr_file):
                st.success("✅ Matrix Calibrated Successfully.")

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
        lbl_passive = st.checkbox("Passive Labels", True)
        label_len = st.slider("Label Length (Words)", min_value=1, max_value=15, value=5, help="Shorten text on the map to reduce clutter. Hover over dots to see full text.")
        
        placeholder_filters = st.container()

        st.divider()
        st.header("⚗️ AI Recommendations")
        enable_clustering = st.checkbox("Show Suggested Clusters (AI Boss)", False) 
        if enable_clustering:
            num_clusters = st.slider("Suggested # of Mindsets", 2, 8, 4)
            strictness = st.slider("🎯 AI Cluster Tightness", 0, 100, 30)

        st.divider()
        st.header("🔄 View Settings")
        map_rotation = st.slider("Map Rotation", 0, 360, 0, step=90)
        map_height = st.slider("Map Height (Fit to screen)", min_value=500, max_value=1200, value=650, step=50, help="Adjust this so the entire map fits on your screen without scrolling. This keeps the Lasso tool always visible!")

    # --- RENDER ---
    if st.session_state.processed_data:
        safe_univ = get_safe_universe()
        fig = go.Figure()
        
        df_b = rotate_coords(st.session_state.df_brands.copy(), map_rotation)
        df_a = rotate_coords(st.session_state.df_attrs.copy(), map_rotation)
        p_source = st.session_state.passive_data if 'passive_data' in st.session_state else []
        df_p_list = [rotate_coords(l.copy(), map_rotation) for l in p_source if isinstance(l, pd.DataFrame) and 'x' in l.columns]
        
        if enable_clustering and HAS_SKLEARN:
            dfs_to_cluster = [df_a[['x', 'y']]]
            for l in df_p_list:
                if not l.empty: dfs_to_cluster.append(l[['x', 'y']])
            pool = pd.concat(dfs_to_cluster)
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(pool)
            df_a['Cluster'] = kmeans.predict(df_a[['x', 'y']])
            df_b['Cluster'] = kmeans.predict(df_b[['x', 'y']]) 
            for l in df_p_list:
                if not l.empty: l['Cluster'] = kmeans.predict(l[['x', 'y']])
        else:
            df_a['Cluster'] = 0; df_b['Cluster'] = 0
            for l in df_p_list:
                if not l.empty: l['Cluster'] = 0

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

        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#1f77b4'), name="Columns"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#d62728'), name="Base Rows"))
        for l in df_p_list:
            if not l.empty:
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#555', symbol=l['Shape'].iloc[0]), name=l['LayerName'].iloc[0]))

        def get_so(lbl, base_c):
            if f_brand == "None": return (base_c, 0.9) 
            return (base_c, 1.0) if (lbl == f_brand or lbl in hl) else ('#d3d3d3', 0.2)

        hl = []
        if f_brand != "None":
            hero = df_b[df_b['Label'] == f_brand].iloc[0]
            for d in [df_a] + df_p_list:
                d['temp_d'] = np.sqrt((d['x']-hero['x'])**2 + (d['y']-hero['y'])**2)
                hl += d.sort_values('temp_d').head(5)['Label'].tolist()

        ai_colors = px.colors.qualitative.Bold

        if show_base_cols:
            if enable_clustering:
                for cid in sorted(df_b['Cluster'].unique()):
                    sub = df_b[df_b['Cluster'] == cid]
                    bc = ai_colors[cid % 10]
                    res = [get_so(r['Label'], bc) for _,r in sub.iterrows()]
                    fig.add_trace(go.Scatter(
                        x=sub['x'], y=sub['y'], 
                        mode='markers', 
                        marker=dict(size=14, color=[r[0] for r in res], opacity=[r[1] for r in res], line=dict(width=1, color='white')), 
                        text=sub['Label'], customdata=sub['Label'], hovertemplate="<b>%{customdata}</b><extra></extra>", showlegend=False
                    ))
                    if lbl_cols:
                        for _, r in sub.iterrows():
                            color, opac = get_so(r['Label'], bc)
                            short_lbl = truncate_label(r['Label'], label_len)
                            fig.add_annotation(x=r['x'], y=r['y'], text=short_lbl, showarrow=False, yshift=10, font=dict(color=bc, size=12))
            else:
                bc = '#1f77b4'
                res = [get_so(r['Label'], bc) for _,r in df_b.iterrows()]
                fig.add_trace(go.Scatter(
                    x=df_b['x'], y=df_b['y'], mode='markers', 
                    marker=dict(size=14, color=[r[0] for r in res], opacity=[r[1] for r in res], line=dict(width=1, color='white')), 
                    text=df_b['Label'], customdata=df_b['Label'], hovertemplate="<b>%{customdata}</b><extra></extra>", showlegend=False
                ))
                if lbl_cols:
                    for _, r
