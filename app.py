import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import pickle
import re
import math 
import textwrap

# --- SAFE IMPORTS FOR MATH & MEANING ---
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_NLP = True
except ImportError:
    HAS_NLP = False

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="The Consumer Landscape")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Quicksand', sans-serif; }
        h1, h2, h3 { font-family: 'Quicksand', sans-serif; font-weight: 700; }
        .stMetric { font-family: 'Quicksand', sans-serif; }
        
        .mindset-card {
            padding: 20px; border-radius: 10px; border-left: 10px solid #ccc;
            background-color: #f9f9f9; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .custom-mindset-card {
            padding: 20px; border-radius: 10px; border-left: 10px solid #673ab7;
            background-color: #f3e5f5; margin-bottom: 25px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .size-badge { float: right; background: #004d40; padding: 4px 10px; border-radius: 20px; font-size: 0.85em; font-weight: 700; color: #fff; }
        .code-block { background-color: #1e1e1e; color: #d4d4d4; padding: 25px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 1.1em; margin-top: 10px; white-space: pre-wrap; border: 1px solid #444; line-height: 1.6; }
        .logic-tag { background: #333; color: #fff; padding: 4px 12px; border-radius: 4px; font-size: 0.9em; font-weight: 700; margin-bottom: 15px; display: inline-block; }
        .passive-tag { background: #e3f2fd; border: 1px solid #90caf9; color: #1565c0; padding: 4px 12px; border-radius: 15px; font-size: 0.85em; font-weight: 700; margin-right: 5px; margin-bottom: 5px; display: inline-block; }
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

if 'df_brands_master' not in st.session_state: st.session_state.df_brands_master = pd.DataFrame()
if 'df_attrs_master' not in st.session_state: st.session_state.df_attrs_master = pd.DataFrame()
if 'max_dim' not in st.session_state: st.session_state.max_dim = 2
if 's_vals' not in st.session_state: st.session_state.s_vals = []

if 'df_brands' not in st.session_state: st.session_state.df_brands = pd.DataFrame()
if 'df_attrs' not in st.session_state: st.session_state.df_attrs = pd.DataFrame()
if 'accuracy' not in st.session_state: st.session_state.accuracy = 0
if 'lasso_labels' not in st.session_state: st.session_state.lasso_labels = []
if 'map_key' not in st.session_state: st.session_state.map_key = 0 
if 'corr_matrix' not in st.session_state: st.session_state.corr_matrix = pd.DataFrame()
if 'saved_mindsets' not in st.session_state: st.session_state.saved_mindsets = []
if 'hidden_export_items' not in st.session_state: st.session_state.hidden_export_items = []

# --- HELPERS ---
@st.cache_resource(show_spinner="Loading Semantic NLP Engine (first time only)...")
def load_nlp_model():
    if HAS_NLP:
        return SentenceTransformer('all-MiniLM-L6-v2')
    return None

nlp_model = load_nlp_model()

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
    
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.set_index(label_col)
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
    
    valid_cols = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if "study universe" in cl or cl in ["total", "base", "sample", "grand total"]:
            continue
        valid_cols.append(c)
        
    mask = []
    for r in df.index:
        rl = str(r).strip().lower()
        if "study universe" in rl or rl in ["total", "base", "sample", "grand total"]:
            mask.append(False)
        else:
            mask.append(True)

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
    
    indep_prob = sum(math.comb(N, k) * (avg_p**k) * ((1 - avg_p)**(N - k)) for k in range(K, N + 1))
    indep_reach = indep_prob * universe_size
    return (overlap_factor * perfect_reach) + ((1.0 - overlap_factor) * indep_reach)

def get_safe_universe():
    return st.session_state.universe_size if st.session_state.universe_size > 0 else 258000.0

def get_weight_lookup():
    wl = dict(zip(st.session_state.df_attrs['Label'], st.session_state.df_attrs['Weight']))
    for layer in st.session_state.passive_data:
        if isinstance(layer, pd.DataFrame) and not layer.empty and 'Label' in layer.columns and 'Weight' in layer.columns:
            wl.update(dict(zip(layer['Label'], layer['Weight'])))
    return wl

# ==========================================
# DATA PROCESSING ENGINE
# ==========================================
def process_data(uploaded_file, passive_files, passive_configs):
    try:
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
            
            # Save ALL axes up to 5 max
            max_dim = min(5, len(s))
            st.session_state.max_dim = max_dim
            st.session_state.s_vals = s
            
            df_b_m = pd.DataFrame(col_coords[:, :max_dim], columns=[f'Dim{i+1}' for i in range(max_dim)])
            df_b_m['Label'] = df_math.columns
            st.session_state.df_brands_master = df_b_m
            
            df_a_m = pd.DataFrame(row_coords[:, :max_dim], columns=[f'Dim{i+1}' for i in range(max_dim)])
            df_a_m['Label'] = df_math.index
            df_a_m['Weight'] = df_math.sum(axis=1).values 
            st.session_state.df_attrs_master = df_a_m

            # Default backwards compatibility to Axis 1 and 2
            st.session_state.df_brands = df_b_m.copy()
            st.session_state.df_brands['x'] = df_b_m['Dim1']
            st.session_state.df_brands['y'] = df_b_m['Dim2'] if max_dim > 1 else df_b_m['Dim1']

            st.session_state.df_attrs = df_a_m.copy()
            st.session_state.df_attrs['x'] = df_a_m['Dim1']
            st.session_state.df_attrs['y'] = df_a_m['Dim2'] if max_dim > 1 else df_a_m['Dim1']
            
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
                pf_bytes = pf.getvalue()
                p_raw = pd.read_csv(io.BytesIO(pf_bytes)) if pf.name.endswith('.csv') else pd.read_excel(io.BytesIO(pf_bytes))
                p_c = clean_df(p_raw, is_core=False)
                
                p_cols_norm = normalize_strings(p_c.columns)
                p_idx_norm = normalize_strings(p_c.index)
                common_b_count = sum(1 for x in p_cols_norm if x in col_mapper)
                common_r_count = sum(1 for x in p_idx_norm if x in row_mapper)
                
                is_rows = (cfg["mode"] == "Rows (Stars)")
                
                proj = np.array([]); p_aligned = pd.DataFrame()
                status_msg = "❌ No Match"
                
                if is_rows:
                    if common_b_count > 0:
                        status_msg = f"✅ Aligned by {common_b_count} Columns"
                        p_aligned = pd.DataFrame(0.0, index=p_c.index, columns=df_math.columns)
                        for orig_col, norm_col in zip(p_c.columns, p_cols_norm):
                            if norm_col in col_mapper: p_aligned.iloc[:, col_mapper[norm_col]] = p_c[orig_col].values
                        proj = (p_aligned.div(p_aligned.sum(axis=1).replace(0,1), axis=0)).values @ col_coords[:, :max_dim] / s[:max_dim]
                        res = pd.DataFrame(proj, columns=[f'Dim{k+1}' for k in range(max_dim)])
                        res['x'] = res['Dim1']; res['y'] = res['Dim2'] if max_dim > 1 else res['Dim1']
                        res['Label'] = p_c.index; res['Weight'] = p_c.sum(axis=1).values; res['Shape'] = 'star'
                    else:
                        status_msg = "❌ Force 'Rows' failed: No matching columns found."
                else:
                    if common_r_count > 0:
                        status_msg = f"✅ Aligned by {common_r_count} Rows"
                        p_aligned = pd.DataFrame(0.0, index=df_math.index, columns=p_c.columns)
                        for orig_row, norm_row in zip(p_c.index, p_idx_norm):
                            if norm_row in row_mapper: p_aligned.iloc[row_mapper[norm_row], :] = p_c.loc[orig_row].values
                        proj = (p_aligned.div(p_aligned.sum(axis=0).replace(0,1), axis=1)).T.values @ row_coords[:, :max_dim] / s[:max_dim]
                        res = pd.DataFrame(proj, columns=[f'Dim{k+1}' for k in range(max_dim)])
                        res['x'] = res['Dim1']; res['y'] = res['Dim2'] if max_dim > 1 else res['Dim1']
                        res['Label'] = p_c.columns; res['Weight'] = p_c.sum(axis=0).values; res['Shape'] = 'diamond'
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
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Strategic Map", "🧹 MRI Cleaner", "📟 Count Code Editor", "📥 PNG Exporter"])

with tab1:
    st.title("🗺️ The Consumer Landscape")
    
    if st.session_state.processed_data and not st.session_state.exact_universe_found:
        st.warning("⚠️ **Warning:** A 'Study Universe' or 'Total Population' row was not found in your base data. The app is using a default estimate of 258,000 (000s).")
        
    with st.sidebar:
        st.header("📂 Data & Projects")
        with st.expander("💾 Manage Project", expanded=False):
            uploaded_project = st.file_uploader("Load .use file", type=["use"], key="loader")
            if uploaded_project:
                try:
                    p_data = pickle.load(uploaded_project)
                    st.session_state.update(p_data)
                    st.session_state.processed_data = True
                    st.rerun()
                except: st.error("Load failed.")
            if st.session_state.processed_data:
                proj_name = st.text_input("Project Name", "Strategy_Map")
                proj_dict = {
                    'df_brands': st.session_state.df_brands, 'df_attrs': st.session_state.df_attrs, 
                    'passive_data': st.session_state.passive_data, 'accuracy': st.session_state.accuracy, 
                    'universe_size': st.session_state.universe_size, 'exact_universe_found': st.session_state.exact_universe_found,
                    'corr_matrix': st.session_state.corr_matrix, 'saved_mindsets': st.session_state.saved_mindsets,
                    'df_brands_master': st.session_state.df_brands_master, 'df_attrs_master': st.session_state.df_attrs_master,
                    'max_dim': st.session_state.get('max_dim', 2), 's_vals': st.session_state.get('s_vals', [])
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
                    p_mode = st.radio("Map As:", ["Rows (Stars)", "Columns (Diamonds)"], index=0, key=f"mode_{pf.name}")
                    if st.session_state.processed_data and i < len(st.session_state.passive_data):
                        st.caption(f"Status: {st.session_state.passive_data[i].get('Status', 'Pending')}")
                    passive_configs.append({"file": pf, "name": p_name, "show": p_show, "mode": p_mode})

        if uploaded_file: process_data(uploaded_file, passive_files, passive_configs)

        st.divider()
        st.header("🔗 Advanced Calibration")
        st.markdown("<span style='font-size: 0.85em; color: #555;'>Upload a Master MRI Cross-Tab Matrix to unlock automated population sizing.</span>", unsafe_allow_html=True)
        corr_file = st.file_uploader("Upload Master Matrix File", type=["csv", "xlsx", "xls"], key="corr_upload")
        if corr_file:
            if process_correlation_matrix(corr_file): st.success("✅ Matrix Calibrated.")

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
        
        if st.session_state.processed_data and 'max_dim' in st.session_state:
            st.markdown("**Map Dimensions**")
            col_ax1, col_ax2 = st.columns(2)
            with col_ax1: x_dim = st.selectbox("X-Axis", range(1, st.session_state.max_dim + 1), index=0)
            with col_ax2: y_dim = st.selectbox("Y-Axis", range(1, st.session_state.max_dim + 1), index=1 if st.session_state.max_dim > 1 else 0)
            
            # Dynamically set x and y coordinates from master memory
            st.session_state.df_brands['x'] = st.session_state.df_brands_master[f'Dim{x_dim}']
            st.session_state.df_brands['y'] = st.session_state.df_brands_master[f'Dim{y_dim}']
            st.session_state.df_attrs['x'] = st.session_state.df_attrs_master[f'Dim{x_dim}']
            st.session_state.df_attrs['y'] = st.session_state.df_attrs_master[f'Dim{y_dim}']
            
            for i, p in enumerate(st.session_state.passive_data):
                if not p.empty and f'Dim{x_dim}' in p.columns:
                    st.session_state.passive_data[i]['x'] = p[f'Dim{x_dim}']
                    st.session_state.passive_data[i]['y'] = p[f'Dim{y_dim}']
                    
            f_col_highlight = st.selectbox("Highlight Column:", ["None"] + sorted(st.session_state.df_brands['Label'].tolist(), key=str.casefold))
        else:
            f_col_highlight = "None"

        lbl_cols = st.checkbox("Column Labels", True)
        lbl_rows = st.checkbox("Row Labels", True)
        lbl_passive = st.checkbox("Passive Labels", True)
        label_len = st.slider("Label Length (Words)", 1, 15, 5)
        
        placeholder_filters = st.container()

        st.divider()
        st.header("⚗️ AI Boss Generator")
        enable_clustering = st.checkbox("Turn on AI Generator", False) 
        
        use_nlp = False
        math_weight = 1.0
        nlp_weight = 0.0
        
        if enable_clustering:
            if HAS_NLP:
                use_nlp = st.checkbox("🧠 Enable Qualitative AI (Semantic Meaning)", False, help="If activated, the AI will read the text to find conceptually related statements, even if they aren't right next to each other on the map.")
                if use_nlp:
                    blend_val = st.slider("Balancing Tool", min_value=0, max_value=100, value=50, step=5, format="%d%%", help="0% = Pure Map Math. 100% = Pure Text Meaning. 50% = Perfect Blend.")
                    st.caption("👈 Pure Math ——— Pure Meaning 👉")
                    nlp_weight = blend_val / 100.0
                    math_weight = 1.0 - nlp_weight
            else:
                st.markdown("<div class='error-box'>⚠️ <b>Qualitative AI Offline:</b> To group statements by their semantic meaning, run <code>pip install sentence-transformers</code> in your terminal.</div>", unsafe_allow_html=True)
            
            num_clusters = 4
            
            if HAS_SKLEARN and st.session_state.processed_data:
                df_a_raw = st.session_state.df_attrs
                p_source_raw = st.session_state.passive_data if 'passive_data' in st.session_state else []
                
                pool_df = pd.DataFrame()
                frames = [df_a_raw[['Label', 'x', 'y']]]
                for l in p_source_raw:
                    if isinstance(l, pd.DataFrame) and not l.empty and 'x' in l.columns:
                        frames.append(l[['Label', 'x', 'y']])
                pool_df = pd.concat(frames).reset_index(drop=True)
                
                scaler = StandardScaler()
                spatial_coords = scaler.fit_transform(pool_df[['x', 'y']])
                
                if use_nlp and nlp_model:
                    text_embeddings = nlp_model.encode(pool_df['Label'].tolist())
                    semantic_coords = scaler.fit_transform(text_embeddings)
                    
                    spatial_scaled = spatial_coords / np.sqrt(2)
                    semantic_scaled = semantic_coords / np.sqrt(text_embeddings.shape[1])
                    final_matrix = np.hstack((spatial_scaled * math_weight, semantic_scaled * nlp_weight))
                else:
                    final_matrix = spatial_coords

                best_k = 4
                if len(final_matrix) > 3:
                    best_score = -1
                    max_test_k = min(8, len(final_matrix) - 1)
                    for k in range(2, max_test_k + 1):
                        km_test = KMeans(n_clusters=k, random_state=42, n_init=10).fit(final_matrix)
                        score = silhouette_score(final_matrix, km_test.labels_)
                        if score > best_score:
                            best_score = score
                            best_k = k
                    st.markdown(f"<div style='background:#e3f2fd; padding:10px; border-radius:5px; color:#0d47a1; margin-bottom:10px; font-size:0.9em;'>📈 <b>Statistical Insight:</b> The optimal number of mindsets is <b>{best_k}</b>.</div>", unsafe_allow_html=True)
                else:
                    best_k = 2

                num_clusters = st.slider("Suggested # of Mindsets", 2, 8, int(best_k))

                # --- STRICT AUTO-GENERATE ENFORCEMENT ---
                if st.session_state.corr_matrix.empty:
                    st.markdown("<div class='error-box'>❌ <b>Matrix Required:</b> Upload a Master Matrix in the 'Advanced Calibration' sidebar to unlock the Auto-Generator.</div>", unsafe_allow_html=True)
                else:
                    if st.button("⚡ Auto-Generate Deck", use_container_width=True):
                        pool_labels = pool_df['Label'].unique().tolist()
                        norm_pool = [normalize_strings(pd.Series([item])).iloc[0] for item in pool_labels]
                        missing_from_matrix = [orig for orig, norm in zip(pool_labels, norm_pool) if norm not in st.session_state.corr_matrix.index or norm not in st.session_state.corr_matrix.columns]
                        
                        if missing_from_matrix:
                            st.error(f"❌ Auto-Generate Aborted: Your uploaded matrix is missing the following items: {', '.join(missing_from_matrix[:5])}{'...' if len(missing_from_matrix)>5 else ''}. Please update your Master Matrix.")
                        else:
                            km_final = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(final_matrix)
                            cluster_map = dict(zip(pool_df['Label'], km_final.labels_))
                            
                            st.session_state.df_attrs['Cluster'] = st.session_state.df_attrs['Label'].map(cluster_map)
                            st.session_state.df_brands['Cluster'] = 0
                            
                            for idx in range(len(st.session_state.passive_data)):
                                if not st.session_state.passive_data[idx].empty:
                                    st.session_state.passive_data[idx]['Cluster'] = st.session_state.passive_data[idx]['Label'].map(cluster_map)

                            wl = get_weight_lookup()
                            su = get_safe_universe()

                            for cid in range(num_clusters):
                                c_all = [label for label, clus in cluster_map.items() if clus == cid]
                                if len(c_all) == 0: continue

                                norm_items = [normalize_strings(pd.Series([item])).iloc[0] for item in c_all]
                                computed_overlap = 1.0
                                
                                valid_norms = [n for n in norm_items if n in st.session_state.corr_matrix.index and n in st.session_state.corr_matrix.columns]
                                if len(valid_norms) > 1:
                                    sub_matrix = st.session_state.corr_matrix.loc[valid_norms, valid_norms]
                                    mask = np.triu(np.ones(sub_matrix.shape), k=1).astype(bool)
                                    avg_matrix_val = sub_matrix.where(mask).mean().mean()
                                    if pd.notna(avg_matrix_val): computed_overlap = np.clip(avg_matrix_val, 0.0, 1.0)

                                l_num_items = len(c_all)
                                l_min_majority = max(1, (l_num_items // 2) + 1)
                                w_array = [wl.get(item, 0) for item in c_all]
                                
                                target_avg_reach = np.mean(w_array) if w_array else 0
                                target_avg_pct = (target_avg_reach / su) * 100 if su > 0 else 0

                                best_thresh = l_min_majority
                                best_diff = float('inf')
                                best_reach = 0

                                for t in range(l_min_majority, l_num_items + 1):
                                    reach = calculate_clustered_reach(w_array, t, su, computed_overlap)
                                    diff = abs(reach - target_avg_reach) 
                                    if diff < best_diff:
                                        best_diff = diff
                                        best_thresh = t
                                        best_reach = reach

                                best_pct = (best_reach / su) * 100 if su > 0 else 0
                                code = "(" + " + ".join([f"[{r}]" for r in c_all]) + f") >= {best_thresh}"

                                st.session_state.saved_mindsets.append({
                                    "name": f"Auto Segment {cid + 1}",
                                    "items": c_all,
                                    "threshold": best_thresh,
                                    "pop": best_reach,
                                    "pct": best_pct,
                                    "target_avg_pct": target_avg_pct,
                                    "code": code,
                                    "overlap": computed_overlap
                                })
                            st.success("✅ Deck Auto-Generated! Scroll to Tab 3 to view.")

        st.divider()
        st.header("🔄 View Settings")
        map_rotation = st.slider("Map Rotation", 0, 360, 0, step=90)
        # --- INCREASED DEFAULT AND MAX HEIGHT ---
        map_height = st.slider("Map Height (Fit to screen)", 600, 1600, 850, step=50)

    # --- RENDER ---
    if st.session_state.processed_data:
        safe_univ = get_safe_universe()
        fig = go.Figure()
        
        df_b = rotate_coords(st.session_state.df_brands.copy(), map_rotation)
        df_a = rotate_coords(st.session_state.df_attrs.copy(), map_rotation)
        p_source = st.session_state.passive_data if 'passive_data' in st.session_state else []
        df_p_list = [rotate_coords(l.copy(), map_rotation) for l in p_source if isinstance(l, pd.DataFrame) and 'x' in l.columns]
        
        if enable_clustering and HAS_SKLEARN and 'Cluster' not in df_a.columns:
             df_a['Cluster'] = 0
             df_b['Cluster'] = 0
             for l in df_p_list:
                if not l.empty: l['Cluster'] = 0
        elif not enable_clustering:
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
                        sel_p = st.multiselect("Visible Items:", l_opts, default=l_opts, key=f"filter_{i}_{layer['Shape'].iloc[0]}")
                        df_p_list[i] = layer[layer['Label'].isin(sel_p)]

        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#1f77b4'), name="Base Columns"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#d62728'), name="Base Rows"))
        for l in df_p_list:
            if not l.empty:
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#555', symbol=l['Shape'].iloc[0]), name=l['LayerName'].iloc[0]))

        def get_so(lbl, base_c):
            if f_col_highlight == "None": return (base_c, 0.9) 
            return (base_c, 1.0) if (lbl == f_col_highlight or lbl in hl) else ('#d3d3d3', 0.2)

        hl = []
        if f_col_highlight != "None":
            hero = df_b[df_b['Label'] == f_col_highlight].iloc[0]
            for d in [df_a] + df_p_list:
                d['temp_d'] = np.sqrt((d['x']-hero['x'])**2 + (d['y']-hero['y'])**2)
                hl += d.sort_values('temp_d').head(5)['Label'].tolist()

        ai_colors = px.colors.qualitative.Bold

        def plot_layer(df_plot, shape, base_color, show_labels, size=10, is_base_col=False):
            if df_plot.empty: return
            if enable_clustering and 'Cluster' in df_plot.columns:
                for cid in sorted(df_plot['Cluster'].unique()):
                    sub = df_plot[df_plot['Cluster'] == cid]
                    bc = ai_colors[cid % 10] if not is_base_col else '#1f77b4' 
                    res = [get_so(r['Label'], bc) for _, r in sub.iterrows()]
                    fig.add_trace(go.Scatter(
                        x=sub['x'], y=sub['y'], mode='markers',
                        marker=dict(size=size, symbol=shape, color=[r[0] for r in res], opacity=[r[1] for r in res], line=dict(width=1, color='white')),
                        text=sub['Label'], customdata=sub['Label'], hovertemplate="<b>%{customdata}</b><extra></extra>", showlegend=False
                    ))
                    if show_labels:
                        for _, r in sub.iterrows():
                            color, opac = get_so(r['Label'], bc)
                            if opac > 0.3 or is_base_col:
                                short_lbl = truncate_label(r['Label'], label_len)
                                fig.add_annotation(x=r['x'], y=r['y'], text=short_lbl, showarrow=False, yshift=10 if is_base_col else 8, font=dict(color=bc, size=12 if is_base_col else 10))
            else:
                res = [get_so(r['Label'], base_color) for _, r in df_plot.iterrows()]
                fig.add_trace(go.Scatter(
                    x=df_plot['x'], y=df_plot['y'], mode='markers',
                    marker=dict(size=size, symbol=shape, color=[r[0] for r in res], opacity=[r[1] for r in res], line=dict(width=1, color='white')),
                    text=df_plot['Label'], customdata=df_plot['Label'], hovertemplate="<b>%{customdata}</b><extra></extra>", showlegend=False
                ))
                if show_labels:
                    for _, r in df_plot.iterrows():
                        color, opac = get_so(r['Label'], base_color)
                        if opac > 0.3 or is_base_col:
                            short_lbl = truncate_label(r['Label'], label_len)
                            fig.add_annotation(x=r['x'], y=r['y'], text=short_lbl, showarrow=False, yshift=10 if is_base_col else 8, font=dict(color=base_color, size=12 if is_base_col else 10))

        if show_base_cols: plot_layer(df_b, 'circle', '#1f77b4', lbl_cols, size=14, is_base_col=True)
        if show_base_rows: plot_layer(df_a, 'circle', '#d62728', lbl_rows, size=10, is_base_col=False)
        for i, layer in enumerate(df_p_list):
            if not layer.empty and layer['Visible'].iloc[0]:
                plot_layer(layer, layer['Shape'].iloc[0], '#555', lbl_passive, size=12, is_base_col=False)

        fig.update_layout(
            template="plotly_white", height=map_height, margin=dict(l=0, r=0, t=30, b=0),
            yaxis_scaleanchor="x", dragmode='pan', hoverlabel=dict(bgcolor="white", font_size=14, font_family="Quicksand"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False), 
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False)
        )
        
        map_event = st.plotly_chart(
            fig, use_container_width=True, on_select="rerun", selection_mode=('lasso', 'box'), 
            key=f"main_map_{st.session_state.map_key}", config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True}
        )
        
        lasso_raw = []
        if map_event and hasattr(map_event, 'selection') and map_event.selection.get("points"):
            for pt in map_event.selection["points"]:
                if "customdata" in pt: lasso_raw.append(pt["customdata"])
                elif "text" in pt: lasso_raw.append(pt["text"])
            
            core_cols = set(st.session_state.df_brands['Label'])
            st.session_state.lasso_labels = [lbl for lbl in lasso_raw if lbl not in core_cols]
        else:
            st.session_state.lasso_labels = []

        if st.session_state.lasso_labels:
            col_msg, col_btn = st.columns([4, 1])
            with col_msg:
                st.markdown(f'<div class="success-box">✅ You bubbled {len(st.session_state.lasso_labels)} statements! Edit them in the Count Code Editor tab.</div>', unsafe_allow_html=True)
            with col_btn:
                if st.button("🗑️ Clear Bubble/Selection", use_container_width=True):
                    st.session_state.lasso_labels = []
                    st.session_state.map_key += 1
                    if "ms_items_lasso" in st.session_state: del st.session_state["ms_items_lasso"]
                    if "ms_thresh_lasso" in st.session_state: del st.session_state["ms_thresh_lasso"]
                    st.rerun()

with tab2:
    st.header("🧹 MRI Data Cleaner")
    st.markdown("Use this tool to clean up messy MRI exports so they are perfectly formatted for the app.")
    
    cleaner_mode = st.radio("What type of data are you cleaning?", [
        "Base Map Data (Columns x Rows)", 
        "Passive Layer Data (Columns x Passives)",
        "Correlation Matrix (Square Crosstab for Calibration)"
    ])
    
    raw_mri = st.file_uploader("Upload Raw Export", type=["csv", "xlsx", "xls"])
    
    if raw_mri:
        try:
            df_r = pd.read_csv(raw_mri, header=None) if raw_mri.name.endswith('.csv') else pd.read_excel(raw_mri, header=None)
            
            if cleaner_mode == "Base Map Data (Columns x Rows)" or cleaner_mode == "Passive Layer Data (Columns x Passives)":
                idx = next(i for i, row in df_r.iterrows() if row.astype(str).str.contains("Weighted \\(000\\)", regex=True).any())
                header_row = df_r.iloc[idx-1].tolist()
                last_val = "Unknown"
                for i in range(len(header_row)):
                    val = str(header_row[i]).strip()
                    if val == "" or val == "nan" or val == "None": header_row[i] = last_val
                    else: last_val = val
                header_r = pd.Series(header_row)
                
                metric_r = df_r.iloc[idx]
                data_r = df_r.iloc[idx+1:].copy()
                
                c_idx, h = [0], ['Attitude']
                for c in range(1, len(metric_r)):
                    if "Weighted" in str(metric_r[c]):
                        h_str = str(header_r[c])
                        if "Universe" not in h_str and h_str.strip().lower() not in ["total", "base"] and h_str != 'nan': 
                            c_idx.append(c); h.append(h_str)
                
                df_c = data_r.iloc[:, c_idx]; df_c.columns = h
                df_c.iloc[:, 0] = df_c.iloc[:, 0].astype(str).str.replace('General Attitudes: ', '', regex=False).str.replace('_Any Agree', '', regex=False).str.replace('"', '', regex=False).str.strip()
                for i in range(1, len(df_c.columns)):
                    df_c.iloc[:, i] = pd.to_numeric(df_c.iloc[:, i].astype(str).str.replace(',', '', regex=False), errors='coerce')
                
                df_c = df_c.loc[:, ~df_c.columns.duplicated()]
                df_c = df_c.dropna(subset=df_c.columns[1:], how='all').fillna(0)
                
                if cleaner_mode == "Base Map Data (Columns x Rows)":
                    st.success("✅ Base Map Data Cleaned!")
                    st.download_button("Download Cleaned Base Data", df_c.to_csv(index=False).encode('utf-8'), "Cleaned_Base_Data.csv")
                else:
                    st.success("✅ Passive Layer Data Cleaned!")
                    st.download_button("Download Cleaned Passive Layer", df_c.to_csv(index=False).encode('utf-8'), "Cleaned_Passive_Layer.csv")

            else:
                idx = next(i for i, row in df_r.iterrows() if row.astype(str).str.contains("%|Percent|Target", case=False, regex=True).any())
                statement_row = df_r.iloc[idx-1].tolist()
                last_val = "Unknown"
                for i in range(len(statement_row)):
                    val = str(statement_row[i]).strip()
                    if val == "" or val == "nan" or val == "None": statement_row[i] = last_val
                    else: last_val = val
                statement_r = pd.Series(statement_row)
                
                metric_r = df_r.iloc[idx]
                data_r = df_r.iloc[idx+1:].copy()
                
                c_idx, h = [0], ['Attitude']
                for c in range(1, len(metric_r)):
                    val = str(metric_r[c]).lower()
                    if "%" in val or "target" in val or "detail" in val or "row" in val or "horiz" in val or "vert" in val:
                        s_str = str(statement_r[c])
                        if "Universe" not in s_str and s_str.strip().lower() not in ["total", "base"] and s_str != 'nan':
                            c_idx.append(c); h.append(s_str)
                
                df_c = data_r.iloc[:, c_idx]; df_c.columns = h
                df_c.iloc[:, 0] = df_c.iloc[:, 0].astype(str).str.replace('General Attitudes: ', '', regex=False).str.replace('_Any Agree', '', regex=False).str.replace('"', '', regex=False).str.strip()
                
                new_cols = ['Attitude']
                for col in df_c.columns[1:]:
                    new_cols.append(str(col).replace('General Attitudes: ', '').replace('_Any Agree', '').replace('"', '').strip())
                df_c.columns = new_cols
                
                for i in range(1, len(df_c.columns)):
                    df_c.iloc[:, i] = df_c.iloc[:, i].astype(str).str.replace(r'[%,]', '', regex=True)
                    df_c.iloc[:, i] = pd.to_numeric(df_c.iloc[:, i], errors='coerce') / 100.0 
                
                df_c = df_c.loc[:, ~df_c.columns.duplicated()]
                df_c = df_c.dropna(subset=df_c.columns[1:], how='all').fillna(0)
                
                st.success("✅ Correlation Matrix Cleaned and Formatted!")
                st.download_button("Download Cleaned Matrix", df_c.to_csv(index=False).encode('utf-8'), "Cleaned_Correlation_Matrix.csv")
        except Exception as e:
            st.error(f"Format error: {e}")

with tab3:
    st.header("📟 Count Code Editor")
    if st.session_state.processed_data and not st.session_state.exact_universe_found:
        st.warning("⚠️ **Warning:** A 'Study Universe' or 'Total Population' row was not found in your base data. The app is using a default estimate of 258,000 (000s).")

    if st.session_state.lasso_labels:
        safe_univ_tab3 = get_safe_universe()
        weight_lookup = get_weight_lookup()
        all_available_labels = sorted(list(weight_lookup.keys()))

        with st.container():
            st.markdown("""<div class="custom-mindset-card">
                <h3 style="color: #4527a0; margin-top:0;">Active Selection Editor</h3>
                <p>Adjust the threshold logic below to refine your audience definition.</p>
            </div>""", unsafe_allow_html=True)
            
            lasso_key = "ms_items_lasso"
            th_lasso_key = "ms_thresh_lasso"
            
            if lasso_key not in st.session_state or set(st.session_state.get('last_lasso_drawn', [])) != set(st.session_state.lasso_labels):
                st.session_state[lasso_key] = st.session_state.lasso_labels
                st.session_state['last_lasso_drawn'] = st.session_state.lasso_labels.copy()
            
            lasso_items = st.multiselect("Selected Statements (Add or Remove):", options=all_available_labels, default=st.session_state[lasso_key], key=lasso_key)
            
            if lasso_items:
                l_num_items = len(lasso_items)
                l_min_majority = max(1, (l_num_items // 2) + 1)
                
                if th_lasso_key not in st.session_state or st.session_state[th_lasso_key] < l_min_majority or st.session_state[th_lasso_key] > l_num_items:
                    st.session_state[th_lasso_key] = l_min_majority

                l_thresh = st.slider("Count Code Threshold (At least X out of Y):", min_value=1, max_value=l_num_items, value=st.session_state[th_lasso_key], key=th_lasso_key)

                norm_items = [normalize_strings(pd.Series([item])).iloc[0] for item in lasso_items]
                missing_items = []
                if not st.session_state.corr_matrix.empty:
                    for orig, norm in zip(lasso_items, norm_items):
                        if norm not in st.session_state.corr_matrix.index or norm not in st.session_state.corr_matrix.columns:
                            missing_items.append(orig)
                else: missing_items = lasso_items
                
                is_calibrated = False
                computed_overlap = 0.0 
                
                if not st.session_state.corr_matrix.empty and len(missing_items) == 0:
                    is_calibrated = True
                    valid_norms = [n for n in norm_items if n in st.session_state.corr_matrix.index and n in st.session_state.corr_matrix.columns]
                    if len(valid_norms) > 1:
                        sub_matrix = st.session_state.corr_matrix.loc[valid_norms, valid_norms]
                        mask = np.triu(np.ones(sub_matrix.shape), k=1).astype(bool)
                        avg_matrix_val = sub_matrix.where(mask).mean().mean()
                        if pd.notna(avg_matrix_val): computed_overlap = np.clip(avg_matrix_val, 0.0, 1.0)
                    elif len(valid_norms) == 1: 
                        computed_overlap = 1.0

                if is_calibrated:
                    st.markdown(f"<div class='calibration-box'>✅ <b>MRI Matrix Calibrated:</b> The exact average pairwise correlation of this custom audience is <b>{computed_overlap*100:.1f}%</b>.</div>", unsafe_allow_html=True)
                    
                    overlap_factor = computed_overlap
                    w_array = [weight_lookup.get(item, 0) for item in lasso_items]
                    l_target_pop = np.mean(w_array) if w_array else 0
                    target_avg_pct = (l_target_pop / safe_univ_tab3) * 100 if safe_univ_tab3 > 0 else 0
                    
                    l_est_reach = calculate_clustered_reach(w_array, l_thresh, safe_univ_tab3, overlap_factor)
                    l_pct = (l_est_reach / safe_univ_tab3) * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        bar_color = '#2e7d32' if l_pct <= 40 else '#c62828'
                        st.markdown(f"""
                            <div style="background-color: #eee; border-radius: 5px; width: 100%; height: 20px;">
                                <div style="background-color: {bar_color}; width: {min(100, l_pct)}%; height: 100%; border-radius: 5px;"></div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                                <span style="font-weight: bold; color: {bar_color}">{l_pct:.1f}% Active Reach</span>
                                <span style="color: #666">Target Average Reference: {target_avg_pct:.1f}%</span>
                            </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"**Active Population Size:** {l_est_reach:,.0f} (000s)")
                        st.markdown(f"*(Target Goal Size: {l_target_pop:,.0f})*")

                    l_code = "(" + " + ".join([f"[{r}]" for r in lasso_items]) + f") >= {l_thresh}"
                    st.markdown(f'<div class="logic-tag">MRI SYNTAX</div><div class="code-block">{l_code}</div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("### 💾 Add to Mindset Deck")
                    col_name, col_save = st.columns([3, 1])
                    with col_name: ms_name = st.text_input("Name this Mindset:", value=f"Custom Audience {len(st.session_state.saved_mindsets) + 1}")
                    with col_save:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("➕ Save Mindset", use_container_width=True):
                            st.session_state.saved_mindsets.append({
                                "name": ms_name, "items": lasso_items, "threshold": l_thresh,
                                "pop": l_est_reach, "pct": l_pct, "code": l_code, "overlap": overlap_factor,
                                "target_avg_pct": target_avg_pct
                            })
                            st.success("Saved! Check your deck below.")
                            st.rerun()
                else:
                    if st.session_state.corr_matrix.empty:
                        st.markdown("<div class='error-box'>❌ <b>Matrix Required:</b> You must upload a Master Correlation Matrix in the 'Advanced Calibration' sidebar to calculate population sizing.</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='error-box'>❌ <b>Data Missing:</b> The following items are missing from your uploaded matrix: <br> `{', '.join(missing_items)}` <br><br> Update your matrix in the sidebar to enable sizing calculations.</div>", unsafe_allow_html=True)
                    
                    l_code = "(" + " + ".join([f"[{r}]" for r in lasso_items]) + f") >= {l_thresh}"
                    st.markdown(f'<div class="logic-tag">MRI SYNTAX (Sizing Locked)</div><div class="code-block">{l_code}</div>', unsafe_allow_html=True)
                    
    else:
        st.info("👈 Open the Map toolbar, select the 'Box Select' (bubble) icon, and draw around points to build a new mindset.")

    st.divider()
    st.header("🗂️ Your Saved Mindsets")
    if not st.session_state.saved_mindsets:
        st.markdown("*Your saved audiences will appear here so you can easily copy and paste them into your strategy deck.*")
    else:
        for idx, ms in enumerate(st.session_state.saved_mindsets):
            with st.expander(f"📦 {ms['name']} — {ms['pct']:.1f}% Reach ({ms['pop']:,.0f}k)", expanded=False):
                st.markdown(f"**Statements Included ({len(ms['items'])}):** {', '.join(ms['items'])}")
                st.markdown(f"**Logic Rule:** Agree with *At Least {ms['threshold']}* of the {len(ms['items'])} statements.")
                st.markdown(f"**Target Average Size:** {ms.get('target_avg_pct', 0):.1f}%")
                st.markdown(f"**MRI Overlap Correlation:** {ms['overlap']*100:.1f}%")
                st.markdown(f'<div class="code-block" style="padding:15px; margin-top:10px; font-size:1em;">{ms["code"]}</div>', unsafe_allow_html=True)
                if st.button("🗑️ Delete Mindset", key=f"del_ms_{idx}"):
                    st.session_state.saved_mindsets.pop(idx)
                    st.rerun()

with tab4:
    st.header("📥 Transparent PNG Exporter")
    st.markdown("Export individual map layers with perfectly locked dimensions so they stack flawlessly on a PowerPoint slide.")

    if st.session_state.processed_data:
        exp_sidebar, exp_main = st.columns([1, 3])

        # 1. Global Axis Locking (Crucial for PPT stacking)
        all_df = [st.session_state.df_brands, st.session_state.df_attrs]
        for p in st.session_state.passive_data:
            if not p.empty and 'x' in p.columns: all_df.append(p)

        x_vals, y_vals = [], []
        for d in all_df:
            x_vals.extend(d['x'].tolist())
            y_vals.extend(d['y'].tolist())

        min_x, max_x = min(x_vals) if x_vals else -1, max(x_vals) if x_vals else 1
        min_y, max_y = min(y_vals) if y_vals else -1, max(y_vals) if y_vals else 1
        
        pad_x = (max_x - min_x) * 0.15 or 1
        pad_y = (max_y - min_y) * 0.15 or 1
        x_range = [min_x - pad_x, max_x + pad_x]
        y_range = [min_y - pad_y, max_y + pad_y]

        with exp_sidebar:
            st.subheader("⚙️ Export Controls")
            
            # Layer Selection
            layer_options = ["Base Columns", "Base Rows"]
            p_names = [l['LayerName'].iloc[0] for l in st.session_state.passive_data if not l.empty and 'Label' in l.columns]
            layer_options.extend(p_names)

            target_layer = st.radio("1. Select Layer to Preview:", layer_options)

            # Get the correct dataframe
            if target_layer == "Base Columns": current_df = st.session_state.df_brands
            elif target_layer == "Base Rows": current_df = st.session_state.df_attrs
            else: current_df = next(l for l in st.session_state.passive_data if not l.empty and l['LayerName'].iloc[0] == target_layer)

            all_labels = current_df['Label'].tolist()

            st.markdown("---")
            st.markdown("**2. Manage Statements**")
            st.caption("Single-click a dot on the map to remove it, or add/remove them from this list manually.")
            
            # --- Robust State Sync ---
            ms_key = f"ms_visible_{target_layer}"
            
            def on_ms_change():
                sel = st.session_state[ms_key]
                hidden_in_layer = [l for l in all_labels if l not in sel]
                hidden_other = [l for l in st.session_state.hidden_export_items if l not in all_labels]
                st.session_state.hidden_export_items = hidden_in_layer + hidden_other

            desired_selections = [lbl for lbl in all_labels if lbl not in st.session_state.hidden_export_items]
            if ms_key not in st.session_state or st.session_state[ms_key] != desired_selections:
                st.session_state[ms_key] = desired_selections

            active_selections = st.multiselect("Visible Items:", options=all_labels, key=ms_key, on_change=on_ms_change)
            # -------------------------
            
            st.markdown("---")
            st.markdown("**3. Visual Settings**")
            
            # Determine default visual settings based on the layer
            def_color = '#1f77b4'
            def_shape = 'circle'
            if target_layer == "Base Rows": 
                def_color = '#d62728'
            elif target_layer not in ["Base Columns", "Base Rows"]:
                def_color = '#555555'
                def_shape = current_df['Shape'].iloc[0] if not current_df.empty and 'Shape' in current_df.columns else 'star'

            shape_options = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star', 'hexagram']
            if def_shape not in shape_options: def_shape = 'circle'

            selected_color = st.color_picker("Layer Color", value=def_color)
            selected_shape = st.selectbox("Layer Shape", options=shape_options, index=shape_options.index(def_shape))
            
            label_position = st.selectbox("Initial Label Position", ["Top", "Bottom", "Left", "Right", "Radial Outward (Auto-Spread)"], index=4)
            tail_length = st.slider("Connector Line Length", 10, 150, 40)
            
            dot_size = st.slider("Dot Size", 4, 30, 10)
            label_size = st.slider("Label Size", 8, 32, 14)
            wrap_length = st.slider("Max Characters Per Line", 10, 150, 40)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Reset All Hidden Statements", use_container_width=True):
                st.session_state.hidden_export_items = []
                st.rerun()

        with exp_main:
            fig_exp = go.Figure()
            annotations = []

            if not current_df.empty:
                # Calculate cluster center for radial spread
                cx = current_df['x'].mean() if not current_df.empty else 0
                cy = current_df['y'].mean() if not current_df.empty else 0

                # Add traces point by point (hidden items are just toggled visible=False to preserve index order)
                for i, row in current_df.iterrows():
                    is_visible = row['Label'] in active_selections
                    wrapped_label = "<br>".join(textwrap.wrap(str(row['Label']), width=wrap_length))
                    
                    # Calculate label starting position offsets (ax, ay)
                    if label_position == "Top": ax_val, ay_val = 0, -tail_length
                    elif label_position == "Bottom": ax_val, ay_val = 0, tail_length
                    elif label_position == "Left": ax_val, ay_val = -tail_length, 0
                    elif label_position == "Right": ax_val, ay_val = tail_length, 0
                    else: # Radial Outward (Auto-Spread)
                        dx = row['x'] - cx
                        dy = row['y'] - cy
                        dist = math.hypot(dx, dy)
                        if dist < 1e-5:
                            ax_val, ay_val = 0, -tail_length
                        else:
                            ax_val = (dx / dist) * tail_length
                            # Plotly ay is pixels down from top, so we reverse it to match math coordinates
                            ay_val = -(dy / dist) * tail_length

                    fig_exp.add_trace(go.Scatter(
                        x=[row['x']], y=[row['y']], mode='markers',
                        marker=dict(size=dot_size, symbol=selected_shape, color=selected_color, line=dict(width=1, color='white')),
                        customdata=[row['Label']],
                        hovertemplate="<b>%{customdata}</b><extra></extra>",
                        name=row['Label'],
                        showlegend=False,
                        visible=True if is_visible else False
                    ))
                    
                    # Convert labels to draggable annotations with connector lines
                    annotations.append(dict(
                        x=row['x'],
                        y=row['y'],
                        xref="x",
                        yref="y",
                        text=wrapped_label,
                        showarrow=True,
                        arrowhead=0,        # Clean line with no arrow head
                        arrowwidth=1,       # Thin connector
                        arrowcolor=selected_color,
                        ax=ax_val,          # Horizontal tail position
                        ay=ay_val,          # Vertical tail position
                        font=dict(size=label_size, color=selected_color, family="Quicksand"),
                        visible=True if is_visible else False
                    ))

            # Lock axes, make background fully transparent, and apply draggable annotations
            fig_exp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                dragmode=False, # Disable panning
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(range=x_range, fixedrange=True, showgrid=False, zeroline=False, visible=False),
                yaxis=dict(range=y_range, fixedrange=True, scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, visible=False),
                annotations=annotations
            )

            # High-Resolution 16:9 Configuration for PowerPoint
            exp_config = {
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d'],
                'edits': {
                    'annotationTail': True,       # Allows dragging the text while keeping the arrow anchored
                    'annotationText': True,       # Allows double-clicking to edit the text itself
                    'annotationPosition': False   # Prevents accidentally moving the anchor away from the dot
                },
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f"{target_layer}_Layer_Export",
                    'height': 720,
                    'width': 1280,
                    'scale': 2 # Multiplies res for crystal clear text
                }
            }

            st.info("📸 **Hover over the top right of the map and click the Camera Icon to download this layer.** You can click and drag labels to reposition them! (Double-click a label to edit its text manually).")

            exp_map_event = st.plotly_chart(
                fig_exp, use_container_width=True, config=exp_config,
                on_select="rerun", selection_mode="points", key=f"exp_map_{target_layer}_{label_size}_{dot_size}_{selected_color}_{selected_shape}_{wrap_length}_{label_position}_{tail_length}"
            )

            # Click-to-hide logic
            if exp_map_event and exp_map_event.selection.get("points"):
                clicked_pts = [pt["customdata"] for pt in exp_map_event.selection["points"] if "customdata" in pt]
                if clicked_pts:
                    changed = False
                    for cp in clicked_pts:
                        if cp not in st.session_state.hidden_export_items:
                            st.session_state.hidden_export_items.append(cp)
                            changed = True
                    if changed:
                        st.rerun()
    else:
        st.info("👈 Upload your Core Data in Tab 1 to unlock the PNG Exporter.")
