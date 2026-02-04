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
            padding: 20px;
            border-radius: 10px;
            border-left: 10px solid #ccc;
            background-color: #f9f9f9;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .size-badge {
            float: right;
            background: #0d47a1;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 800;
            color: #fff;
        }
        .code-block {
            background-color: #1e1e1e;
            color: #d4d4d4;
            padding: 25px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 1.1em;
            margin-top: 10px;
            white-space: pre-wrap;
            border: 1px solid #444;
            line-height: 1.6;
        }
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
    st.session_state.messages = [
        {"role": "assistant", "content": "I am your Landscape Guide. Ask me about **Mindsets**, **White Space**, or specific **Columns**."}
    ]

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
        with st.expander("💾 Manage Project", expanded=False):
            uploaded_project = st.file_uploader("Load .use file", type=["use"], key="loader")
            if uploaded_project is not None:
                try:
                    data = pickle.load(uploaded_project)
                    st.session_state.df_brands = data['df_brands']
                    st.session_state.df_attrs = data['df_attrs']
                    st.session_state.passive_data = data['passive_data']
                    st.session_state.accuracy = data['accuracy']
                    st.session_state.landscape_avg_weight = data.get('landscape_avg_weight', 1.0)
                    st.session_state.processed_data = True
                    st.success("Loaded!")
                    st.rerun()
                except: st.error("Error loading file")
            proj_name = st.text_input("Project Name", "My_Landscape_Map")
            if st.session_state.processed_data:
                project_data = {
                    'df_brands': st.session_state.df_brands,
                    'df_attrs': st.session_state.df_attrs,
                    'passive_data': st.session_state.passive_data,
                    'accuracy': st.session_state.accuracy,
                    'landscape_avg_weight': getattr(st.session_state, 'landscape_avg_weight', 1.0)
                }
                buffer = io.BytesIO(); pickle.dump(project_data, buffer); buffer.seek(0)
                st.download_button("Save Project 📥", buffer, f"{proj_name}.use")

        uploaded_file = st.file_uploader("Upload Core Data", type=["csv", "xlsx", "xls"], key="active")
        passive_files = st.file_uploader("Upload Passive Layers", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="passive")
        
        st.divider()
        st.header("🎨 Layer Visibility")
        col_base_1, col_base_2 = st.columns(2)
        show_base_cols = col_base_1.checkbox("Base Columns", True, key="viz_base_cols")
        show_base_rows = col_base_2.checkbox("Base Rows", True, key="viz_base_rows")
        
        passive_configs = []
        if passive_files:
            st.subheader("Passive Layers")
            for p_file in passive_files:
                with st.expander(f"⚙️ {p_file.name}"):
                    p_name = st.text_input("Label", p_file.name, key=f"name_{p_file.name}")
                    p_show = st.checkbox("Show Layer", True, key=f"show_{p_file.name}")
                    p_mode = st.radio("Map As:", ["Auto", "Rows (Stars)", "Columns (Diamonds)"], key=f"mode_{p_file.name}")
                    passive_configs.append({"file": p_file, "name": p_name, "show": p_show, "mode": p_mode})
        
        st.divider()
        st.header("🏷️ Label Manager")
        lbl_cols = st.checkbox("Show Column Labels", True)
        lbl_rows = st.checkbox("Show Row Labels", True)
        lbl_passive = st.checkbox("Show Passive Labels", True)

        st.divider()
        st.header("⚗️ Mindset Maker")
        enable_clustering = st.checkbox("Enable Mindset Discovery", True)
        num_clusters = 4
        if enable_clustering:
            if HAS_SKLEARN: num_clusters = st.slider("Number of Mindsets", 2, 8, 4)
            else: st.error("Library missing.")
        
        map_rotation = st.slider("🔄 Map Rotation", 0, 360, 0, step=90)
        
        st.divider()
        st.header("🔍 Filter & Highlight")
        placeholder_filters = st.empty()

    if uploaded_file is not None:
        try:
            df_raw = load_file(uploaded_file)
            df_math = clean_df(df_raw)
            df_math = df_math.loc[(df_math != 0).any(axis=1)] 
            df_math = df_math.loc[:, (df_math != 0).any(axis=0)]
            
            if not df_math.empty:
                N = df_math.values; P = N / N.sum(); r = P.sum(axis=1); c = P.sum(axis=0)
                E = np.outer(r, c); E[E < 1e-9] = 1e-9; R = (P - E) / np.sqrt(E)
                U, s, Vh = np.linalg.svd(R, full_matrices=False)
                inertia = s**2; map_accuracy = (np.sum(inertia[:2]) / np.sum(inertia)) * 100
                row_coords = (U * s) / np.sqrt(r[:, np.newaxis])
                col_coords = (Vh.T * s) / np.sqrt(c[:, np.newaxis])
                
                st.session_state.df_brands = pd.DataFrame(col_coords[:, :2], columns=['x', 'y'])
                st.session_state.df_brands['Label'] = df_math.columns
                
                st.session_state.df_attrs = pd.DataFrame(row_coords[:, :2], columns=['x', 'y'])
                st.session_state.df_attrs['Label'] = df_math.index
                st.session_state.df_attrs['Weight'] = df_math.sum(axis=1).values 

                st.session_state.accuracy = map_accuracy
                st.session_state.landscape_avg_weight = df_math.sum(axis=1).mean()
                st.session_state.processed_data = True

                passive_layer_data = []
                for cfg in passive_configs:
                    try:
                        p_df = load_file(cfg["file"]); p_clean = clean_df(p_df)
                        common_brands = list(set(p_clean.columns) & set(df_math.columns))
                        common_attrs = list(set(p_clean.index) & set(df_math.index))
                        is_rows = cfg["mode"] == "Rows (Stars)" if cfg["mode"] != "Auto" else len(common_brands) > len(common_attrs)
                        
                        if is_rows:
                            p_clean = p_clean.loc[[r for r in p_clean.index if r not in df_math.index]]
                            if not p_clean.empty and len(common_brands) > 0:
                                p_aligned = p_clean[common_brands].reindex(columns=df_math.columns).fillna(0)
                                p_prof = p_aligned.div(p_aligned.sum(axis=1).replace(0,1), axis=0)
                                proj = p_prof.values @ col_coords[:, :2] / s[:2]
                                res = pd.DataFrame(proj, columns=['x', 'y'])
                                res['Label'] = p_aligned.index; res['Shape'] = 'star'; res['LayerName'] = cfg["name"]; res['Visible'] = cfg["show"]
                                res['Weight'] = p_clean.sum(axis=1).values
                                passive_layer_data.append(res)
                        else:
                            p_clean = p_clean[[c for c in p_clean.columns if c not in df_math.columns]]
                            if not p_clean.empty and len(common_attrs) > 0:
                                p_aligned = p_clean.reindex(df_math.index).fillna(0)
                                p_prof = p_aligned.div(p_aligned.sum(axis=0).replace(0,1), axis=1)
                                proj = p_prof.T.values @ row_coords[:, :2] / s[:2]
                                res = pd.DataFrame(proj, columns=['x', 'y'])
                                res['Label'] = p_aligned.columns; res['Shape'] = 'diamond'; res['LayerName'] = cfg["name"]; res['Visible'] = cfg["show"]
                                res['Weight'] = p_clean.sum(axis=0).values
                                passive_layer_data.append(res)
                    except: pass
                st.session_state.passive_data = passive_layer_data
        except Exception as e: st.error(f"Error: {e}")

    if st.session_state.processed_data:
        df_brands = rotate_coords(st.session_state.df_brands.copy(), map_rotation)
        df_attrs = rotate_coords(st.session_state.df_attrs.copy(), map_rotation)
        passive_source = st.session_state.passive_data if 'passive_data' in st.session_state else []
        passive_layer_data = [rotate_coords(l.copy(), map_rotation) for l in passive_source]
        
        cluster_colors = px.colors.qualitative.Bold
        
        mindset_report = []
        if enable_clustering and HAS_SKLEARN:
            # 1. Pool all data for co-clustering
            pool_data = df_attrs[['x', 'y']].copy()
            for layer in passive_layer_data:
                pool_data = pd.concat([pool_data, layer[['x', 'y']]])
            
            # 2. Fit K-Means
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(pool_data)
            centroids = kmeans.cluster_centers_
            
            # 3. Predict clusters for all datasets
            df_attrs['Cluster'] = kmeans.predict(df_attrs[['x', 'y']])
            df_brands['Cluster'] = kmeans.predict(df_brands[['x', 'y']])
            for layer in passive_layer_data:
                if not layer.empty: layer['Cluster'] = kmeans.predict(layer[['x', 'y']])
            
            # 4. Generate Report
            for i in range(num_clusters):
                c_actives = df_attrs[df_attrs['Cluster'] == i].copy()
                if not c_actives.empty:
                    c_actives['dist'] = np.sqrt((c_actives['x'] - centroids[i][0])**2 + (c_actives['y'] - centroids[i][1])**2)
                
                c_passives_list = []
                for layer in passive_layer_data:
                    p_match = layer[layer['Cluster'] == i].copy()
                    if not p_match.empty:
                        p_match['dist'] = np.sqrt((p_match['x'] - centroids[i][0])**2 + (p_match['y'] - centroids[i][1])**2)
                        c_passives_list.append(p_match)
                
                c_all_signals = pd.DataFrame()
                if not c_actives.empty: c_all_signals = pd.concat([c_all_signals, c_actives[['Label', 'Weight', 'dist']]])
                if c_passives_list: c_all_signals = pd.concat([c_all_signals, pd.concat(c_passives_list)[['Label', 'Weight', 'dist']]])
                
                if not c_all_signals.empty:
                    sorted_all = c_all_signals.sort_values('dist').drop_duplicates(subset=['Label'])
                    reach_proxy = sorted_all.head(5)['Weight'].mean()
                    avg_weight = getattr(st.session_state, 'landscape_avg_weight', 1.0)
                    if avg_weight == 0: avg_weight = 1.0
                    reach_share = (reach_proxy / avg_weight) * 25
                    
                    mindset_report.append({
                        "id": i+1, "color": cluster_colors[i % len(cluster_colors)], 
                        "rows": sorted_all['Label'].tolist()[:10],
                        "brands": df_brands[df_brands['Cluster'] == i]['Label'].tolist(),
                        "size": reach_share, "threshold": 3 if reach_share > 10 else 2
                    })
        st.session_state.mindset_report = mindset_report

        with placeholder_filters.container():
            st.metric("Landscape Stability", f"{st.session_state.accuracy:.1f}%")
            st.divider()
            focus_brand = st.selectbox("Highlight Column:", ["None"] + sorted(df_brands['Label'].tolist()))
            with st.expander("Filter Base Map"):
                sel_brands = st.multiselect("Columns:", sorted(df_brands['Label'].tolist()), default=df_brands['Label'].tolist()) if not st.checkbox("All Columns", True) else df_brands['Label'].tolist()
                sel_attrs = st.multiselect("Rows:", sorted(df_attrs['Label'].tolist()), default=df_attrs.sort_values('Weight', ascending=False).head(15)['Label'].tolist()) if not st.checkbox("All Rows", True) else df_attrs['Label'].tolist()
            for i, layer in enumerate(passive_layer_data):
                if not layer.empty and layer['Visible'].iloc[0]:
                    with st.expander(f"Filter {layer['LayerName'].iloc[0]}"):
                        l_labels = sorted(layer['Label'].tolist())
                        if not st.checkbox("All Items", True, key=f"f_all_{i}"):
                            passive_layer_data[i] = layer[layer['Label'].isin(st.multiselect("Select:", l_labels, default=l_labels, key=f"f_sel_{i}"))]

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
                    l_copy = layer.copy(); l_copy['D'] = np.sqrt((l_copy['x']-hx)**2 + (l_copy['y']-hy)**2)
                    hl += l_copy.sort_values('D').head(5)['Label'].tolist()

        def get_so(lbl, base_c):
            if focus_brand == "None": return base_c, 0.9
            if lbl == focus_brand or lbl in hl: return base_c, 1.0
            return '#d3d3d3', 0.2

        # Brands
        if show_base_cols:
            plot_b = df_brands[df_brands['Label'].isin(sel_brands)]
            res = [get_so(r['Label'], '#1f77b4') for _, r in plot_b.iterrows()]
            fig.add_trace(go.Scatter(
                x=plot_b['x'], y=plot_b['y'], mode='markers',
                marker=dict(size=10, color=[r[0] for r in res], opacity=[r[1] for r in res], line=dict(width=1, color='white')),
                text=plot_b['Label'], hoverinfo='text', name='Columns'
            ))
            if lbl_cols:
                for _, r in plot_b.iterrows():
                    c, o = get_so(r['Label'], '#1f77b4')
                    if o > 0.4: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=c, ax=0, ay=-15, font=dict(color=c, size=11))

        # Mindsets (Active Rows)
        if show_base_rows:
            plot_a = df_attrs[df_attrs['Label'].isin(sel_attrs)]
            if enable_clustering:
                for cid in sorted(plot_a['Cluster'].unique()):
                    sub = plot_a[plot_a['Cluster'] == cid]; bc = cluster_colors[cid % len(cluster_colors)]
                    res = [get_so(r['Label'], bc) for _, r in sub.iterrows()]
                    fig.add_trace(go.Scatter(
                        x=sub['x'], y=sub['y'], mode='markers',
                        marker=dict(size=7, color=[r[0] for r in res], opacity=[r[1] for r in res]),
                        text=sub['Label'], hoverinfo='text', name=f"Mindset {cid+1}"
                    ))
                    if lbl_rows:
                        for _, r in sub.iterrows():
                            c, o = get_so(r['Label'], bc)
                            if o > 0.4: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=c, ax=0, ay=-15, font=dict(color=c, size=11))
            else:
                res = [get_so(r['Label'], '#d62728') for _, r in plot_a.iterrows()]
                fig.add_trace(go.Scatter(
                    x=plot_a['x'], y=plot_a['y'], mode='markers',
                    marker=dict(size=7, color=[r[0] for r in res], opacity=[r[1] for r in res]),
                    text=plot_a['Label'], hoverinfo='text', name='Base Rows'
                ))
                if lbl_rows:
                    for _, r in plot_a.iterrows():
                        c, o = get_so(r['Label'], '#d62728')
                        if o > 0.4: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=c, ax=0, ay=-15, font=dict(color=c, size=11))

        # Passives (Colored by Cluster)
        for i, layer in enumerate(passive_layer_data):
            if not layer.empty and layer['Visible'].iloc[0]:
                
                # Assign colors point-by-point based on the Unified Cluster ID
                p_colors = []
                p_opacities = []
                
                for _, r in layer.iterrows():
                    # Get base color from the Cluster ID
                    if enable_clustering and 'Cluster' in layer.columns:
                        cid = int(r['Cluster'])
                        base_c = cluster_colors[cid % len(cluster_colors)]
                    else:
                        base_c = cluster_colors[i % len(cluster_colors)]
                    
                    c, o = get_so(r['Label'], base_c)
                    p_colors.append(c)
                    p_opacities.append(o)

                fig.add_trace(go.Scatter(
                    x=layer['x'], y=layer['y'], mode='markers',
                    marker=dict(size=9, symbol=layer['Shape'].iloc[0], color=p_colors, opacity=p_opacities, line=dict(width=1, color='white')),
                    text=layer['Label'], hoverinfo='text', name=layer['LayerName'].iloc[0]
                ))
                
                if lbl_passive:
                    for idx, r in layer.iterrows():
                        # Retrieve the specific color calculated for this point
                        color_for_text = p_colors[idx] if idx < len(p_colors) else '#333'
                        opacity_for_text = p_opacities[idx] if idx < len(p_opacities) else 1.0
                        
                        if opacity_for_text > 0.4:
                            fig.add_annotation(
                                x=r['x'], y=r['y'], text=r['Label'],
                                showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=color_for_text,
                                ax=0, ay=-15, font=dict(color=color_for_text, size=10)
                            )

        fig.update_layout(title={'text': "Strategic Map", 'y':0.95, 'x':0.5, 'xanchor':'center', 'font': {'family': 'Nunito', 'size': 20}}, template="plotly_white", height=850, xaxis=dict(showgrid=False, showticklabels=False, zeroline=True), yaxis=dict(showgrid=False, showticklabels=False, zeroline=True), yaxis_scaleanchor="x", yaxis_scaleratio=1, dragmode='pan')
        st.plotly_chart(fig, use_container_width=True, config={'editable': True, 'scrollZoom': True})

        if enable_clustering and mindset_report:
            st.divider(); st.header("⚗️ Strategic Mindset Briefing")
            cols = st.columns(min(3, num_clusters))
            for i, t in enumerate(mindset_report):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="mindset-card" style="border-left-color: {t['color']};">
                        <span class="size-badge">~{t['size']:.1f}% Reach Est.</span>
                        <h3 style="color: {t['color']}; margin-top:0;">Mindset {t['id']}</h3>
                        <p><b>Defining Rows:</b><br>{", ".join(t['rows'][:5])}...</p>
                        <p><b>Involved Columns:</b><br>{", ".join(t['brands']) if t['brands'] else "<i>None.</i>"}</p>
                    </div>
                    """, unsafe_allow_html=True)

# ==========================================
# TAB 2: AI CHAT
# ==========================================
with tab2:
    st.header("💬 AI Landscape Chat")
    if not st.session_state.processed_data: st.warning("👈 Upload data first.")
    else:
        def analyze_query(query):
            q, df_b, df_a = query.lower(), st.session_state.df_brands, st.session_state.df_attrs
            if "theme" in q: return f"**Themes:**\n* ↔️ **X-Axis:** {df_a.loc[df_a['x'].idxmin()]['Label']} to {df_a.loc[df_a['x'].idxmax()]['Label']}\n* ↕️ **Y-Axis:** {df_a.loc[df_a['y'].idxmin()]['Label']} to {df_a.loc[df_a['y'].idxmax()]['Label']}"
            for b in df_b['Label']:
                if b.lower() in q:
                    br = df_b[df_b['Label']==b].iloc[0]; df_a['D'] = np.sqrt((df_a['x']-br['x'])**2 + (df_a['y']-br['y'])**2)
                    return f"**Audit: {b}**\n✅ **Strengths:** {', '.join(df_a.sort_values('D').head(3)['Label'].tolist())}"
            return "Ask about Themes or Columns."
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        if p := st.chat_input("Ask..."):
            st.session_state.messages.append({"role": "user", "content": p})
            with st.chat_message("user"): st.markdown(p)
            r = analyze_query(p); st.session_state.messages.append({"role": "assistant", "content": r})
            with st.chat_message("assistant"): st.markdown(r)

# ==========================================
# TAB 3: CLEANER
# ==========================================
with tab3:
    st.header("🧹 MRI Data Cleaner")
    raw_mri = st.file_uploader("Upload Raw MRI", type=["csv", "xlsx", "xls"])
    if raw_mri:
        try:
            df_raw = pd.read_csv(raw_mri, header=None) if raw_mri.name.endswith('.csv') else pd.read_excel(raw_mri, header=None)
            idx = next(i for i, row in df_raw.iterrows() if row.astype(str).str.contains("Weighted (000)", regex=False).any())
            brand_row = df_raw.iloc[idx - 1]; metric_row = df_raw.iloc[idx]; data_rows = df_raw.iloc[idx+1:].copy()
            cols, headers = [0], ['Attitude']
            for c in range(1, len(metric_row)):
                if "Weighted" in str(metric_row[c]):
                    brand = str(brand_row[c-1])
                    if "Study Universe" not in brand and "Total" not in brand and brand != 'nan':
                        cols.append(c); headers.append(brand)
            df_clean = data_rows.iloc[:, cols]; df_clean.columns = headers
            df_clean['Attitude'] = df_clean['Attitude'].astype(str).str.replace('General Attitudes: ', '', regex=False)
            for c in df_clean.columns[1:]: df_clean[c] = pd.to_numeric(df_clean[c].astype(str).str.replace(',', ''), errors='coerce')
            df_clean = df_clean.dropna(subset=df_clean.columns[1:], how='all')
            df_clean = df_clean[df_clean[df_clean.columns[1:]].fillna(0).sum(axis=1) > 0]
            df_clean = df_clean[df_clean['Attitude'].str.len() > 3]
            df_clean = df_clean[~df_clean['Attitude'].astype(str).str.contains("Study Universe|Total|Base|Sample", case=False, regex=True)]
            st.success("Cleaned!"); st.download_button("Download CSV", df_clean.to_csv(index=False).encode('utf-8'), "Cleaned_MRI.csv", "text/csv")
        except: st.error("Invalid MRI format.")

# ==========================================
# TAB 4: 📟 COUNT CODE MAKER
# ==========================================
with tab4:
    st.header("📟 Count Code Maker")
    st.markdown("Automated targeting optimized for unique reach across all map signals.")
    if not st.session_state.processed_data or not st.session_state.mindset_report:
        st.warning("⚠️ Turn on Mindset Discovery to generate targets.")
    else:
        for t in st.session_state.mindset_report:
            with st.expander(f"Mindset {t['id']} Unified Target (Est. {t['size']:.1f}% Reach)", expanded=True):
                rows = t['rows']
                threshold = t['threshold']
                mri_code = "(" + " + ".join([f"[{r}]" for r in rows]) + f") >= {threshold}"
                st.markdown('<div class="logic-tag">MRI-SIMMONS TARGET CODE</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="code-block">{mri_code}</div>', unsafe_allow_html=True)
                st.info(f"Targets the top signals from all layers. Calibrated for a high-signal {t['size']:.1f}% audience.")
