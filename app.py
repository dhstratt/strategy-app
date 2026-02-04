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
    from sklearn.metrics import pairwise_distances_argmin_min
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
        .size-badge { float: right; background: #eee; padding: 2px 8px; border-radius: 20px; font-size: 0.8em; font-weight: 700; color: #666; }
        .code-block { background-color: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 0.9em; margin-top: 10px; white-space: pre-wrap; word-wrap: break-word; }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = False
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am your Landscape Guide."}]

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
    df['x'], df['y'] = rotated[:, 0], rotated[:, 1]
    return df

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
                    st.session_state.processed_data = True
                    st.rerun()
                except: st.error("Error loading file")
            proj_name = st.text_input("Project Name", "My_Landscape_Map")
            if st.session_state.processed_data:
                project_data = {
                    'df_brands': st.session_state.df_brands, 
                    'df_attrs': st.session_state.df_attrs, 
                    'passive_data': st.session_state.passive_data, 
                    'accuracy': st.session_state.accuracy
                }
                buffer = io.BytesIO()
                pickle.dump(project_data, buffer)
                buffer.seek(0)
                st.download_button("Save Project 📥", buffer, f"{proj_name}.use")

        uploaded_file = st.file_uploader("Upload Core Data", type=["csv", "xlsx", "xls"], key="active")
        passive_files = st.file_uploader("Upload Passive Layers", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="passive")
        st.divider()
        st.header("🎨 Layer Visibility")
        show_base_cols = st.checkbox("Base Columns", True, key="viz_base_cols")
        show_base_rows = st.checkbox("Base Rows", True, key="viz_base_rows")
        passive_configs = []
        if passive_files:
            for p_file in passive_files:
                with st.expander(f"⚙️ {p_file.name}"):
                    passive_configs.append({
                        "file": p_file, "name": st.text_input("Label", p_file.name, key=f"n_{p_file.name}"),
                        "show": st.checkbox("Show", True, key=f"s_{p_file.name}"),
                        "mode": st.radio("Map As:", ["Auto", "Rows (Stars)", "Columns (Diamonds)"], key=f"m_{p_file.name}")
                    })
        st.divider()
        st.header("⚗️ Mindset Maker")
        enable_clustering = st.checkbox("Enable Mindset Discovery", False)
        num_clusters = st.slider("Number of Mindsets", 2, 8, 4) if enable_clustering else 4
        map_rotation = st.slider("🔄 Map Rotation", 0, 360, 0, step=90)
        st.divider()
        placeholder_filters = st.empty()

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
                st.session_state.accuracy = (np.sum(s**2[:2]) / np.sum(s**2)) * 100
                st.session_state.processed_data = True
                
                passive_layer_data = []
                for cfg in passive_configs:
                    p_clean = clean_df(load_file(cfg["file"]))
                    common_brands = list(set(p_clean.columns) & set(df_math.columns))
                    common_attrs = list(set(p_clean.index) & set(df_math.index))
                    is_rows = cfg["mode"] == "Rows (Stars)" if cfg["mode"] != "Auto" else len(common_brands) > len(common_attrs)
                    if is_rows:
                        p_aligned = p_clean[common_brands].reindex(columns=df_math.columns).fillna(0)
                        proj = (p_aligned.div(p_aligned.sum(axis=1).replace(0,1), axis=0)).values @ col_coords[:, :2] / s[:2]
                        res = pd.DataFrame(proj, columns=['x', 'y']); res['Label'] = p_aligned.index; res['Shape'] = 'star'; res['LayerName'] = cfg["name"]; res['Visible'] = cfg["show"]; res['Weight'] = p_clean.sum(axis=1).values
                        passive_layer_data.append(res)
                    else:
                        p_aligned = p_clean.reindex(df_math.index).fillna(0)
                        proj = p_aligned.div(p_aligned.sum(axis=0).replace(0,1), axis=1).T.values @ row_coords[:, :2] / s[:2]
                        res = pd.DataFrame(proj, columns=['x', 'y']); res['Label'] = p_clean.columns; res['Shape'] = 'diamond'; res['LayerName'] = cfg["name"]; res['Visible'] = cfg["show"]; res['Weight'] = p_clean.sum(axis=0).values
                        passive_layer_data.append(res)
                st.session_state.passive_data = passive_layer_data
        except Exception as e: st.error(f"Error: {e}")

    if st.session_state.processed_data:
        df_brands = rotate_coords(st.session_state.df_brands.copy(), map_rotation)
        df_attrs = rotate_coords(st.session_state.df_attrs.copy(), map_rotation)
        passive_layer_data = [rotate_coords(l.copy(), map_rotation) for l in st.session_state.passive_data]
        cluster_colors = px.colors.qualitative.Bold
        mindset_report = []
        
        # Calculate Total Weight safely from the attributes dataframe
        total_weight_denominator = df_attrs['Weight'].sum() if 'Weight' in df_attrs.columns else 1.0

        if enable_clustering and HAS_SKLEARN:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            df_attrs['Cluster'] = kmeans.fit_predict(df_attrs[['x', 'y']])
            centroids = kmeans.cluster_centers_
            df_brands['Cluster'] = kmeans.predict(df_brands[['x', 'y']])
            for i in range(num_clusters):
                c_rows = df_attrs[df_attrs['Cluster'] == i].copy()
                c_rows['dist'] = np.sqrt((c_rows['x'] - centroids[i][0])**2 + (c_rows['y'] - centroids[i][1])**2)
                mindset_report.append({
                    "id": i+1, 
                    "color": cluster_colors[i % len(cluster_colors)], 
                    "rows": c_rows.sort_values('dist').head(10)['Label'].tolist(), 
                    "full_rows": c_rows['Label'].tolist(), 
                    "brands": df_brands[df_brands['Cluster'] == i]['Label'].tolist(), 
                    "size": (c_rows['Weight'].sum() / total_weight_denominator)*100
                })
            for layer in passive_layer_data: 
                if not layer.empty: layer['Cluster'] = kmeans.predict(layer[['x', 'y']])
        
        with placeholder_filters.container():
            st.metric("Landscape Stability", f"{st.session_state.accuracy:.1f}%")
            focus_brand = st.selectbox("Highlight Column:", ["None"] + sorted(df_brands['Label'].tolist()))
            with st.expander("Filter Base Map"):
                sel_brands = st.multiselect("Columns:", sorted(df_brands['Label'].tolist()), default=df_brands['Label'].tolist()) if not st.checkbox("All Columns", True) else df_brands['Label'].tolist()
                sel_attrs = st.multiselect("Rows:", sorted(df_attrs['Label'].tolist()), default=df_attrs.sort_values('Weight', ascending=False).head(15)['Label'].tolist()) if not st.checkbox("All Rows", True) else df_attrs['Label'].tolist()
            
        fig = go.Figure()
        hl = []
        if focus_brand != "None":
            hero = df_brands[df_brands['Label'] == focus_brand].iloc[0]
            hx, hy = hero['x'], hero['y']
            active_attrs = df_attrs[df_attrs['Label'].isin(sel_attrs)].copy()
            active_attrs['D'] = np.sqrt((active_attrs['x']-hx)**2 + (active_attrs['y']-hy)**2)
            hl += active_attrs.sort_values('D').head(5)['Label'].tolist()
            for layer in passive_layer_data:
                if not layer.empty and layer['Visible'].iloc[0]:
                    l_copy = layer.copy()
                    l_copy['D'] = np.sqrt((l_copy['x']-hx)**2 + (l_copy['y']-hy)**2)
                    hl += l_copy.sort_values('D').head(5)['Label'].tolist()

        def get_so(lbl, base_c):
            if focus_brand == "None": return base_c, 0.9
            return (base_c, 1.0) if (lbl == focus_brand or lbl in hl) else ('#d3d3d3', 0.25)

        if show_base_cols:
            plot_b = df_brands[df_brands['Label'].isin(sel_brands)]
            res = [get_so(r['Label'], '#1f77b4') for _, r in plot_b.iterrows()]
            fig.add_trace(go.Scatter(x=plot_b['x'], y=plot_b['y'], mode='markers', marker=dict(size=10, color=[r[0] for r in res], opacity=[r[1] for r in res], line=dict(width=1, color='white')), text=plot_b['Label'], name='Columns'))
            for _, r in plot_b.iterrows():
                c, o = get_so(r['Label'], '#1f77b4')
                if o > 0.4: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], ax=0, ay=-20, font=dict(color=c, size=11), arrowcolor=c)

        if show_base_rows:
            plot_a = df_attrs[df_attrs['Label'].isin(sel_attrs)]
            if enable_clustering:
                for cid in sorted(plot_a['Cluster'].unique()):
                    sub = plot_a[plot_a['Cluster'] == cid]; bc = cluster_colors[cid % len(cluster_colors)]
                    res = [get_so(r['Label'], bc) for _, r in sub.iterrows()]
                    fig.add_trace(go.Scatter(x=sub['x'], y=sub['y'], mode='markers', marker=dict(size=7, color=[r[0] for r in res], opacity=[r[1] for r in res]), text=sub['Label'], name=f"Mindset {cid+1}"))
                    for _, r in sub.iterrows():
                        c, o = get_so(r['Label'], bc)
                        if o > 0.4: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], ax=0, ay=-15, font=dict(color=c, size=11), arrowcolor=c)
            else:
                res = [get_so(r['Label'], '#d62728') for _, r in plot_a.iterrows()]
                fig.add_trace(go.Scatter(x=plot_a['x'], y=plot_a['y'], mode='markers', marker=dict(size=7, color=[r[0] for r in res], opacity=[r[1] for r in res]), text=plot_a['Label'], name='Base Rows'))
                for _, r in plot_a.iterrows():
                    c, o = get_so(r['Label'], '#d62728')
                    if o > 0.4: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], ax=0, ay=-15, font=dict(color=c, size=11), arrowcolor=c)

        for i, layer in enumerate(passive_layer_data):
            if not layer.empty and layer['Visible'].iloc[0]:
                if enable_clustering and 'Cluster' in layer.columns:
                    for cid in sorted(layer['Cluster'].unique()):
                        sub = layer[layer['Cluster'] == cid]; bc = cluster_colors[cid % len(cluster_colors)]
                        res = [get_so(r['Label'], bc) for _, r in sub.iterrows()]
                        fig.add_trace(go.Scatter(x=sub['x'], y=sub['y'], mode='markers', marker=dict(size=9, symbol=sub['Shape'].iloc[0], color=[r[0] for r in res], opacity=[r[1] for r in res], line=dict(width=1, color='white')), text=sub['Label'], name=f"{sub['LayerName'].iloc[0]} (M{cid+1})", showlegend=False))
                else:
                    bc = cluster_colors[i % len(cluster_colors)]
                    res = [get_so(r['Label'], bc) for _, r in layer.iterrows()]
                    fig.add_trace(go.Scatter(x=layer['x'], y=layer['y'], mode='markers', marker=dict(size=9, symbol=layer['Shape'].iloc[0], color=[r[0] for r in res], opacity=[r[1] for r in res], line=dict(width=1, color='white')), text=layer['Label'], name=layer['LayerName'].iloc[0]))
        
        fig.update_layout(title={'text': "Strategic Map", 'y':0.95, 'x':0.5, 'xanchor':'center'}, template="plotly_white", height=850, yaxis_scaleanchor="x", dragmode='pan')
        st.plotly_chart(fig, use_container_width=True, config={'editable': True, 'scrollZoom': True})

        if enable_clustering and mindset_report:
            st.divider(); st.header("⚗️ Strategic Mindset Briefing")
            cols = st.columns(min(3, num_clusters))
            for i, t in enumerate(mindset_report):
                with cols[i % 3]:
                    st.markdown(f'<div class="mindset-card" style="border-left-color: {t["color"]};"><span class="size-badge">{t["size"]:.1f}% volume</span><h3 style="color: {t["color"]}; margin-top:0;">Mindset {t["id"]}</h3><p><b>Core:</b><br>{", ".join(t["rows"][:5])}</p><p><b>Brands:</b><br>{", ".join(t["brands"][:5])}</p></div>', unsafe_allow_html=True)

# ==========================================
# TAB 2 & 3
# ==========================================
with tab2: st.header("💬 AI Chat"); st.write("Use the Landscape Guide.")
with tab3: st.header("🧹 Cleaner"); st.write("Upload raw MRI data.")

# ==========================================
# TAB 4:📟 COUNT CODE MAKER
# ==========================================
with tab4:
    st.header("📟 Count Code Maker")
    st.markdown("Copy these strings directly into **MRI-Simmons** to build custom audiences for each mindset.")
    
    if not st.session_state.processed_data or not enable_clustering:
        st.warning("⚠️ Turn on **Mindset Discovery** in the map sidebar to generate codes.")
    else:
        for t in mindset_report:
            with st.expander(f"Mindset {t['id']} Target Code ({t['size']:.1f}% Reach)", expanded=True):
                st.markdown(f"**Description:** This target is defined by individuals who agree with the following **{len(t['full_rows'])}** statements.")
                mri_string = " OR ".join([f"[{r}]" for r in t['full_rows']])
                st.markdown(f'<div class="code-block">{mri_string}</div>', unsafe_allow_html=True)
                st.info(f"💡 **Math Check:** Summed Weighted Volume for these rows / Total Map Volume = **{t['size']:.2f}%**")
