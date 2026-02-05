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
        .size-badge { float: right; background: #004d40; padding: 4px 10px; border-radius: 20px; font-size: 0.85em; font-weight: 800; color: #fff; }
        .code-block { background-color: #1e1e1e; color: #d4d4d4; padding: 25px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 1.1em; margin-top: 10px; white-space: pre-wrap; border: 1px solid #444; line-height: 1.6; }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = False
if 'passive_data' not in st.session_state:
    st.session_state.passive_data = [] 
if 'mindset_report' not in st.session_state:
    st.session_state.mindset_report = []
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am your Landscape Guide. Ask me about **Mindsets**, **White Space**, or specific **Columns**."}]

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
    
    # NEW: Capture Universe Size for Population Scaling
    universe_row = df[df.index.astype(str).str.contains("Study Universe|Total Population", case=False, regex=True)]
    if not universe_row.empty:
        st.session_state.universe_size = universe_row.iloc[0].sum()
    else:
        st.session_state.universe_size = 258000 # Fallback for MRI Adults
        
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
        with st.expander("💾 Manage Project", expanded=False):
            uploaded_project = st.file_uploader("Load .use file", type=["use"], key="loader")
            if uploaded_project is not None:
                try:
                    data = pickle.load(uploaded_project)
                    st.session_state.df_brands = data['df_brands']
                    st.session_state.df_attrs = data['df_attrs']
                    st.session_state.passive_data = data['passive_data']
                    st.session_state.accuracy = data['accuracy']
                    st.session_state.universe_size = data.get('universe_size', 258000)
                    st.session_state.processed_data = True
                    st.rerun()
                except: st.error("Error loading file")
            proj_name = st.text_input("Project Name", "My_Map")
            if st.session_state.processed_data:
                proj_data = {'df_brands': st.session_state.df_brands, 'df_attrs': st.session_state.df_attrs, 'passive_data': st.session_state.passive_data, 'accuracy': st.session_state.accuracy, 'universe_size': st.session_state.universe_size}
                buffer = io.BytesIO(); pickle.dump(proj_data, buffer); buffer.seek(0)
                st.download_button("Save Project 📥", buffer, f"{proj_name}.use")

        uploaded_file = st.file_uploader("Upload Core Data", type=["csv", "xlsx", "xls"], key="active")
        passive_files = st.file_uploader("Upload Passive Layers", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="passive")
        
        st.divider()
        st.header("🏷️ Label Manager")
        lbl_cols = st.checkbox("Show Column Labels", True)
        lbl_rows = st.checkbox("Show Row Labels", True)
        lbl_passive = st.checkbox("Show Passive Labels", True)

        st.divider()
        st.header("⚗️ Mindset Precision")
        enable_clustering = st.checkbox("Enable Mindset Discovery", True)
        num_clusters = st.slider("Number of Mindsets", 2, 8, 4)
        strictness = st.slider("🎯 Definition Tightness", 0, 100, 30)
        map_rotation = st.slider("🔄 Map Rotation", 0, 360, 0, step=90)
        st.divider()
        placeholder_filters = st.empty()

    if uploaded_file is not None:
        try:
            df_math = clean_df(load_file(uploaded_file))
            if not df_math.empty:
                N = df_math.values; P = N/N.sum(); r = P.sum(axis=1); c = P.sum(axis=0); E = np.outer(r,c)
                E[E<1e-9]=1e-9; R=(P-E)/np.sqrt(E); U, s, Vh = np.linalg.svd(R, full_matrices=False)
                row_coords = (U*s)/np.sqrt(r[:, np.newaxis]); col_coords = (Vh.T*s)/np.sqrt(c[:, np.newaxis])
                st.session_state.df_brands = pd.DataFrame(col_coords[:, :2], columns=['x','y']); st.session_state.df_brands['Label'] = df_math.columns
                st.session_state.df_attrs = pd.DataFrame(row_coords[:, :2], columns=['x','y']); st.session_state.df_attrs['Label'] = df_math.index
                st.session_state.df_attrs['Weight'] = df_math.sum(axis=1).values 
                st.session_state.accuracy = (np.sum(s**2[:2])/np.sum(s**2))*100
                st.session_state.processed_data = True

                # Process Passives (Preserving exact logic)
                passive_list = []
                for pf in (passive_files or []):
                    p_clean = clean_df(load_file(pf))
                    common = list(set(p_clean.columns) & set(df_math.columns))
                    if common:
                        p_aligned = p_clean[common].reindex(columns=df_math.columns).fillna(0)
                        proj = (p_aligned.div(p_aligned.sum(axis=1).replace(0,1), axis=0)).values @ col_coords[:,:2] / s[:2]
                        res = pd.DataFrame(proj, columns=['x','y']); res['Label'] = p_aligned.index; res['Weight'] = p_clean.sum(axis=1).values
                        res['LayerName'] = pf.name; res['Shape'] = 'star'; res['Visible'] = True
                        passive_list.append(res)
                st.session_state.passive_data = passive_list
        except Exception as e: st.error(f"Error: {e}")

    if st.session_state.processed_data:
        df_b = rotate_coords(st.session_state.df_brands.copy(), map_rotation)
        df_a = rotate_coords(st.session_state.df_attrs.copy(), map_rotation)
        pass_source = st.session_state.passive_data if 'passive_data' in st.session_state else []
        df_p_list = [rotate_coords(l.copy(), map_rotation) for l in pass_source]
        
        mindset_report = []
        if enable_clustering and HAS_SKLEARN:
            pool = pd.concat([df_a[['x','y']]] + [l[['x','y']] for l in df_p_list])
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(pool)
            centroids = kmeans.cluster_centers_
            df_a['Cluster'] = kmeans.predict(df_a[['x','y']])
            df_b['Cluster'] = kmeans.predict(df_b[['x','y']])
            for l in df_p_list: l['Cluster'] = kmeans.predict(l[['x','y']])
            
            for i in range(num_clusters):
                c_actives = df_a[df_a['Cluster'] == i].copy()
                c_actives['dist'] = np.sqrt((c_actives['x']-centroids[i][0])**2 + (c_actives['y']-centroids[i][1])**2)
                
                c_passives = pd.concat([l[l['Cluster'] == i] for l in df_p_list]) if df_p_list else pd.DataFrame()
                if not c_passives.empty:
                    c_passives['dist'] = np.sqrt((c_passives['x']-centroids[i][0])**2 + (c_passives['y']-centroids[i][1])**2)
                
                all_cluster_sigs = pd.concat([c_actives[['Label','Weight','dist']], c_passives[['Label','Weight','dist']] if not c_passives.empty else None])
                dists = all_cluster_sigs['dist'].tolist()
                cutoff = np.percentile(dists, 100 - strictness)
                
                df_a.loc[df_a['Cluster']==i, 'IsCore'] = c_actives['dist'] <= cutoff
                for l in df_p_list: l.loc[l['Cluster']==i, 'IsCore'] = (np.sqrt((l['x']-centroids[i][0])**2 + (l['y']-centroids[i][1])**2)) <= cutoff
                
                core_sigs = all_cluster_sigs[all_cluster_sigs['dist'] <= cutoff].sort_values('dist').drop_duplicates('Label')
                pop_weight = core_sigs.head(5)['Weight'].mean()
                pop_pct = (pop_weight / st.session_state.universe_size) * 100
                
                mindset_report.append({"id": i+1, "color": px.colors.qualitative.Bold[i % 10], "rows": core_sigs['Label'].tolist()[:10], "pop_000s": pop_weight, "percent": pop_pct, "brands": df_b[df_b['Cluster']==i]['Label'].tolist(), "threshold": 3 if pop_pct > 12 else 2})
        
        st.session_state.mindset_report = mindset_report
        with placeholder_filters.container():
            col1, col2 = st.columns([1, 2])
            col1.metric("Stability", f"{st.session_state.accuracy:.1f}%")
            view_mode = col2.selectbox("👁️ View Focus:", ["Show All"] + [f"Mindset {m['id']}" for m in mindset_report])
            focus_brand = st.selectbox("Highlight Column:", ["None"] + sorted(df_b['Label'].tolist()))

        fig = go.Figure()
        def get_so(lbl, base_c, is_core=True):
            if focus_brand == "None": return (base_c, 0.9) if is_core else ('#eeeeee', 0.15)
            return (base_c, 1.0) if lbl == focus_brand or lbl in hl else ('#d3d3d3', 0.2)

        hl = []
        if focus_brand != "None":
            hero = df_b[df_b['Label'] == focus_brand].iloc[0]
            for d in [df_a] + df_p_list:
                d['temp_d'] = np.sqrt((d['x']-hero['x'])**2 + (d['y']-hero['y'])**2)
                hl += d.sort_values('temp_d').head(5)['Label'].tolist()

        if show_base_cols:
            res = [get_so(r['Label'], '#1f77b4') for _,r in df_b.iterrows()]
            fig.add_trace(go.Scatter(x=df_b['x'], y=df_b['y'], mode='markers', marker=dict(size=12, color=[r[0] for r in res], opacity=[r[1] for r in res], line=dict(width=1, color='white')), text=df_b['Label'], name='Brands'))
            if lbl_cols:
                for _, r in df_b.iterrows():
                    c, o = get_so(r['Label'], '#1f77b4'); fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], showarrow=True, arrowhead=0, arrowcolor=c, ax=0, ay=-20, font=dict(color=c, size=12))

        if show_base_rows:
            for cid in sorted(df_a['Cluster'].unique()):
                if view_mode != "Show All" and view_mode != f"Mindset {cid+1}": continue
                sub = df_a[df_a['Cluster'] == cid]; bc = px.colors.qualitative.Bold[cid % 10]
                res = [get_so(r['Label'], bc, r.get('IsCore', True)) for _,r in sub.iterrows()]
                fig.add_trace(go.Scatter(x=sub['x'], y=sub['y'], mode='markers', marker=dict(size=8, color=[r[0] for r in res], opacity=[r[1] for r in res]), text=sub['Label'], name=f"Mindset {cid+1}"))
                if lbl_rows:
                    for _, r in sub.iterrows():
                        c, o = get_so(r['Label'], bc, r.get('IsCore', True))
                        if o > 0.3: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], showarrow=True, arrowhead=0, arrowcolor=c, ax=0, ay=-15, font=dict(color=c, size=11))

        for i, layer in enumerate(df_p_list):
            if not layer.empty:
                for cid in sorted(layer['Cluster'].unique()):
                    if view_mode != "Show All" and view_mode != f"Mindset {cid+1}": continue
                    sub = layer[layer['Cluster'] == cid]; bc = px.colors.qualitative.Bold[cid % 10]
                    res = [get_so(r['Label'], bc, r.get('IsCore', True)) for _,r in sub.iterrows()]
                    fig.add_trace(go.Scatter(x=sub['x'], y=sub['y'], mode='markers', marker=dict(size=10, symbol='diamond', color=[r[0] for r in res], opacity=[r[1] for r in res], line=dict(width=1, color='white')), text=sub['Label'], name=layer['LayerName'].iloc[0]))
                    if lbl_passive:
                        for _, r in sub.iterrows():
                            c, o = get_so(r['Label'], bc, r.get('IsCore', True))
                            if o > 0.3: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], showarrow=True, arrowhead=0, arrowcolor=c, ax=0, ay=-15, font=dict(color=c, size=10))

        fig.update_layout(template="plotly_white", height=850, yaxis_scaleanchor="x", dragmode='pan')
        st.plotly_chart(fig, use_container_width=True, config={'editable': True, 'scrollZoom': True})

        if mindset_report:
            st.divider(); st.header("👥 Population Mindsets")
            cols = st.columns(3)
            for i, m in enumerate(mindset_report):
                with cols[i % 3]:
                    st.markdown(f'<div class="mindset-card" style="border-left-color: {m["color"]};"><span class="size-badge">{m["percent"]:.1f}% Pop.</span><h3 style="color: {m["color"]};">{m["id"]}</h3><p><b>Top Signals:</b> {", ".join(m["rows"][:3])}</p></div>', unsafe_allow_html=True)

# AI, Cleaner, and Count Code tabs follow exactly as before...
