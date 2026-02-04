import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import collections
import io
import pickle

# --- SAFE IMPORT FOR CLUSTERING ---
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
        .size-badge { float: right; background: #0d47a1; padding: 4px 10px; border-radius: 20px; font-size: 0.85em; font-weight: 800; color: #fff; }
        .code-block { background-color: #1e1e1e; color: #d4d4d4; padding: 25px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 1.1em; margin-top: 10px; white-space: pre-wrap; border: 1px solid #444; line-height: 1.6; }
        .logic-tag { background: #333; color: #fff; padding: 4px 12px; border-radius: 4px; font-size: 0.9em; font-weight: 800; margin-bottom: 15px; display: inline-block; }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = False
if 'mindset_report' not in st.session_state:
    st.session_state.mindset_report = []

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ The Consumer Landscape", "💬 AI Landscape Chat", "🧹 MRI Data Cleaner", "📟 Count Code Maker"])

# --- HELPERS ---
def clean_df(df):
    label_col = df.columns[0]
    df = df.set_index(label_col)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    df = df[~df.index.astype(str).str.contains("Study Universe|Total|Base|Sample", case=False, regex=True)]
    valid_cols = [c for c in df.columns if "study universe" not in str(c).lower() and "total" not in str(c).lower() and "base" not in str(c).lower()]
    return df[valid_cols]

def load_file(file):
    if file.name.endswith('.csv'): return pd.read_csv(file)
    else: return pd.read_excel(file)

def rotate_coords(df, angle_deg):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    coords = df[['x', 'y']].values
    rotated = coords @ R.T
    df_new = df.copy()
    df_new['x'], df_new['y'] = rotated[:, 0], rotated[:, 1]
    return df_new

# ==========================================
# TAB 1: THE CONSUMER LANDSCAPE
# ==========================================
with tab1:
    st.title("🗺️ The Consumer Landscape")
    with st.sidebar:
        st.header("📂 Data & Projects")
        uploaded_file = st.file_uploader("Upload Core Data", type=["csv", "xlsx", "xls"], key="active")
        passive_files = st.file_uploader("Upload Passive Layers", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="passive")
        
        st.divider()
        st.header("🏷️ Label Manager")
        lbl_cols = st.checkbox("Show Column Labels", True)
        lbl_rows = st.checkbox("Show Row Labels", False)
        lbl_passive = st.checkbox("Show Passive Labels", True)

        st.divider()
        st.header("⚗️ Mindset Maker")
        enable_clustering = st.checkbox("Enable Mindset Discovery", True)
        num_clusters = st.slider("Number of Mindsets", 2, 8, 4)
        map_rotation = st.slider("🔄 Map Rotation", 0, 360, 0, step=90)
        st.divider()
        placeholder_stats = st.empty()

    if uploaded_file is not None:
        try:
            df_math_raw = clean_df(load_file(uploaded_file))
            df_math = df_math_raw.loc[(df_math_raw != 0).any(axis=1)]
            if not df_math.empty:
                N = df_math.values; P = N / N.sum(); r = P.sum(axis=1); c = P.sum(axis=0); E = np.outer(r, c)
                E[E < 1e-9] = 1e-9; R = (P - E) / np.sqrt(E); U, s, Vh = np.linalg.svd(R, full_matrices=False)
                row_coords = (U * s) / np.sqrt(r[:, np.newaxis]); col_coords = (Vh.T * s) / np.sqrt(c[:, np.newaxis])
                
                st.session_state.df_brands = pd.DataFrame(col_coords[:, :2], columns=['x', 'y'])
                st.session_state.df_brands['Label'] = df_math.columns
                
                st.session_state.df_attrs = pd.DataFrame(row_coords[:, :2], columns=['x', 'y'])
                st.session_state.df_attrs['Label'] = df_math.index
                st.session_state.df_attrs['Weight'] = df_math.sum(axis=1).values
                
                inertia = s**2
                st.session_state.accuracy = (np.sum(inertia[:2]) / np.sum(inertia)) * 100
                st.session_state.landscape_avg_weight = df_math.sum(axis=1).mean()
                st.session_state.processed_data = True

                integrated_passives = []
                for p_file in (passive_files or []):
                    p_clean = clean_df(load_file(p_file))
                    common_brands = list(set(p_clean.columns) & set(df_math.columns))
                    if len(common_brands) > 0:
                        # De-duplicate: If statement exists in core, skip it in passive
                        p_aligned = p_clean[common_brands].reindex(columns=df_math.columns).fillna(0)
                        unique_passive_rows = [r for r in p_aligned.index if r not in df_math.index]
                        if unique_passive_rows:
                            p_final = p_aligned.loc[unique_passive_rows]
                            proj = (p_final.div(p_final.sum(axis=1).replace(0,1), axis=0)).values @ col_coords[:, :2] / s[:2]
                            res = pd.DataFrame(proj, columns=['x', 'y']); res['Label'] = p_final.index
                            res['Weight'] = p_clean.loc[unique_passive_rows].sum(axis=1).values 
                            integrated_passives.append(res)
                st.session_state.passive_data = integrated_passives

        except Exception as e: st.error(f"Error: {e}")

    if st.session_state.processed_data:
        df_brands = rotate_coords(st.session_state.df_brands.copy(), map_rotation)
        df_attrs = rotate_coords(st.session_state.df_attrs.copy(), map_rotation)
        cluster_colors = px.colors.qualitative.Bold
        mindset_report = []
        
        with placeholder_stats.container():
            st.metric("Landscape Stability", f"{st.session_state.accuracy:.1f}%")
            if st.session_state.accuracy < 60: st.warning("⚠️ Low Stability")

        if enable_clustering and HAS_SKLEARN:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            df_attrs['Cluster'] = kmeans.fit_predict(df_attrs[['x', 'y']])
            centroids = kmeans.cluster_centers_
            
            for i in range(num_clusters):
                c_actives = df_attrs[df_attrs['Cluster'] == i].copy()
                c_actives['dist'] = np.sqrt((c_actives['x'] - centroids[i][0])**2 + (c_actives['y'] - centroids[i][1])**2)
                
                c_passives_list = []
                for layer in st.session_state.passive_data:
                    l_rot = rotate_coords(layer.copy(), map_rotation)
                    l_rot['Cluster'] = kmeans.predict(l_rot[['x', 'y']])
                    p_match = l_rot[l_rot['Cluster'] == i].copy()
                    if not p_match.empty:
                        p_match['dist'] = np.sqrt((p_match['x'] - centroids[i][0])**2 + (p_match['y'] - centroids[i][1])**2)
                        c_passives_list.append(p_match)
                
                if c_passives_list:
                    c_all_signals = pd.concat([c_actives[['Label', 'Weight', 'dist']] , pd.concat(c_passives_list)[['Label', 'Weight', 'dist']]])
                else:
                    c_all_signals = c_actives[['Label', 'Weight', 'dist']]
                
                sorted_all = c_all_signals.sort_values('dist').drop_duplicates(subset=['Label'])
                reach_proxy = sorted_all.head(5)['Weight'].mean()
                reach_share = (reach_proxy / st.session_state.landscape_avg_weight) * 25 
                
                mindset_report.append({
                    "id": i+1, "color": cluster_colors[i % len(cluster_colors)], 
                    "top_rows": sorted_all['Label'].tolist()[:10],
                    "size": reach_share, "threshold": 3 if reach_share > 10 else 2
                })
        st.session_state.mindset_report = mindset_report
        
        fig = go.Figure()
        
        # Columns
        fig.add_trace(go.Scatter(
            x=df_brands['x'], y=df_brands['y'], mode='markers+text' if lbl_cols else 'markers',
            marker=dict(size=10, color='#1f77b4', line=dict(width=1, color='white')),
            text=df_brands['Label'], textposition="top center", name='Brands'
        ))
        
        # Rows (Mindsets)
        for cid in sorted(df_attrs['Cluster'].unique()):
            sub = df_attrs[df_attrs['Cluster'] == cid]; bc = cluster_colors[cid % len(cluster_colors)]
            fig.add_trace(go.Scatter(
                x=sub['x'], y=sub['y'], mode='markers+text' if lbl_rows else 'markers',
                marker=dict(size=7, color=bc, opacity=0.7),
                text=sub['Label'], textposition="top center", name=f"Mindset {cid+1}"
            ))
            
        # Passives
        for layer in st.session_state.passive_data:
            l_rot = rotate_coords(layer.copy(), map_rotation)
            fig.add_trace(go.Scatter(
                x=l_rot['x'], y=l_rot['y'], mode='markers+text' if lbl_passive else 'markers',
                marker=dict(size=8, symbol='diamond', color='rgba(0,0,0,0)', line=dict(width=1, color='#333')),
                text=l_rot['Label'], textposition="top center", name='Passive Items'
            ))
        
        fig.update_layout(template="plotly_white", height=800, yaxis_scaleanchor="x", dragmode='pan')
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 3: 🧹 CLEANER (RESTORED)
# ==========================================
with tab3:
    st.header("🧹 MRI Data Cleaner")
    raw_mri = st.file_uploader("Upload Raw MRI Export", type=["csv", "xlsx", "xls"])
    if raw_mri:
        try:
            df_raw = pd.read_csv(raw_mri, header=None) if raw_mri.name.endswith('.csv') else pd.read_excel(raw_mri, header=None)
            idx = next(i for i, row in df_raw.iterrows() if "Weighted (000)" in str(row.values))
            brand_row = df_raw.iloc[idx - 1]; metric_row = df_raw.iloc[idx]; data_rows = df_raw.iloc[idx+1:].copy()
            cols, h = [0], ['Attitude']
            for c in range(1, len(metric_row)):
                if "Weighted" in str(metric_row[c]):
                    b = str(brand_row[c-1])
                    if b != 'nan' and "Total" not in b: cols.append(c); h.append(b)
            df_c = data_rows.iloc[:, cols]; df_c.columns = h
            df_c['Attitude'] = df_c['Attitude'].astype(str).str.replace('General Attitudes: ', '')
            for c in df_c.columns[1:]: df_c[c] = pd.to_numeric(df_c[c].astype(str).str.replace(',', ''), errors='coerce')
            df_c = df_c.dropna(subset=df_c.columns[1:], how='all')
            st.success("Cleaned!"); st.download_button("Download CSV", df_c.to_csv(index=False).encode('utf-8'), "Cleaned_MRI.csv")
        except: st.error("Error cleaning file.")

# ==========================================
# TAB 4: 📟 COUNT CODE MAKER
# ==========================================
with tab4:
    st.header("📟 Count Code Maker")
    if not st.session_state.mindset_report:
        st.warning("⚠️ Discovery required.")
    else:
        for t in st.session_state.mindset_report:
            with st.expander(f"Mindset {t['id']} Unified Target ({t['size']:.1f}% Reach)", expanded=True):
                rows = t['top_rows']
                mri_code = "(" + " + ".join([f"[{r}]" for r in rows]) + f") >= {t['threshold']}"
                st.markdown(f'<div class="code-block">{mri_code}</div>', unsafe_allow_html=True)
