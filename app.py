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

# --- CUSTOM CSS: NUNITO FONT & CARDS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
        h1, h2, h3 { font-family: 'Nunito', sans-serif; font-weight: 800; }
        .stMetric { font-family: 'Nunito', sans-serif; }
        .territory-card {
            padding: 20px;
            border-radius: 10px;
            border-left: 10px solid #ccc;
            background-color: #f9f9f9;
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = False
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am your Landscape Guide. Ask me about **Themes**, **White Space**, or specific **Columns**."}
    ]

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🗺️ The Consumer Landscape", "💬 AI Landscape Chat", "🧹 MRI Data Cleaner"])

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

# ==========================================
# TAB 1: THE CONSUMER LANDSCAPE
# ==========================================
with tab1:
    st.title("🗺️ The Consumer Landscape")
    
    # --- SIDEBAR ---
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
                    st.success("Loaded!")
                    st.rerun()
                except: st.error("Error loading file")
            proj_name = st.text_input("Project Name", "My_Map")
            if st.session_state.processed_data:
                project_data = {'df_brands': st.session_state.df_brands, 'df_attrs': st.session_state.df_attrs, 'passive_data': st.session_state.passive_data, 'accuracy': st.session_state.accuracy}
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
        st.header("⚗️ Analysis")
        enable_clustering = st.checkbox("Enable Territory Discovery", False)
        num_clusters = 4
        if enable_clustering:
            if HAS_SKLEARN: num_clusters = st.slider("Number of Territories", 2, 8, 4)
            else: st.error("ML Library missing.")
        
        st.divider()
        st.header("🔍 Filter & Highlight")
        placeholder_filters = st.empty()

    # --- PROCESSING ---
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
                
                df_brands = pd.DataFrame(col_coords[:, :2], columns=['x', 'y'])
                df_brands['Label'] = df_math.columns; df_brands['Type'] = 'Column'
                df_brands['Distinctiveness'] = np.sqrt(df_brands['x']**2 + df_brands['y']**2)
                
                df_attrs = pd.DataFrame(row_coords[:, :2], columns=['x', 'y'])
                df_attrs['Label'] = df_math.index; df_attrs['Type'] = 'Row'
                df_attrs['Distinctiveness'] = np.sqrt(df_attrs['x']**2 + df_attrs['y']**2)

                st.session_state.processed_data = True
                st.session_state.df_brands = df_brands
                st.session_state.df_attrs = df_attrs
                st.session_state.accuracy = map_accuracy

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
                                p_prof = p_clean[common_brands].div(p_clean[common_brands].sum(axis=1).replace(0,1), axis=0)
                                proj = p_prof.values @ col_coords[:, :2] / s[:2]
                                res = pd.DataFrame(proj, columns=['x', 'y'])
                                res['Label'] = p_clean.index; res['Shape'] = 'star'; res['LayerName'] = cfg["name"]; res['Visible'] = cfg["show"]
                                res['Distinctiveness'] = np.sqrt(res['x']**2 + res['y']**2)
                                passive_layer_data.append(res)
                        else:
                            p_clean = p_clean[[c for c in p_clean.columns if c not in df_math.columns]]
                            if not p_clean.empty and len(common_attrs) > 0:
                                p_prof = p_clean.reindex(df_math.index).div(p_clean.reindex(df_math.index).sum(axis=0).replace(0,1), axis=1)
                                proj = p_prof.T.values @ row_coords[:, :2] / s[:2]
                                res = pd.DataFrame(proj, columns=['x', 'y'])
                                res['Label'] = p_clean.columns; res['Shape'] = 'diamond'; res['LayerName'] = cfg["name"]; res['Visible'] = cfg["show"]
                                res['Distinctiveness'] = np.sqrt(res['x']**2 + res['y']**2)
                                passive_layer_data.append(res)
                    except: pass
                st.session_state.passive_data = passive_layer_data
        except Exception as e: st.error(f"Error: {e}")

    # --- RENDER ---
    if st.session_state.processed_data:
        df_brands = st.session_state.df_brands; df_attrs = st.session_state.df_attrs
        passive_layer_data = st.session_state.passive_data
        cluster_colors = px.colors.qualitative.Bold
        
        # 1. Clustering Logic
        territory_report = []
        if enable_clustering and HAS_SKLEARN:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            df_attrs['Cluster'] = kmeans.fit_predict(df_attrs[['x', 'y']])
            centroids = kmeans.cluster_centers_
            df_brands['Cluster'] = kmeans.predict(df_brands[['x', 'y']])
            
            for i in range(num_clusters):
                cluster_rows = df_attrs[df_attrs['Cluster'] == i].copy()
                cluster_rows['dist'] = np.sqrt((cluster_rows['x'] - centroids[i][0])**2 + (cluster_rows['y'] - centroids[i][1])**2)
                top_rows = cluster_rows.sort_values('dist').head(5)['Label'].tolist()
                top_brands = df_brands[df_brands['Cluster'] == i]['Label'].tolist()
                territory_report.append({"id": i+1, "color": cluster_colors[i % len(cluster_colors)], "rows": top_rows, "brands": top_brands})
                
            for layer in passive_layer_data:
                if not layer.empty: layer['Cluster'] = kmeans.predict(layer[['x', 'y']])

        # 2. Filters UI
        with placeholder_filters.container():
            # STABILITY METRIC (RESTORED)
            st.metric("Map Stability Score", f"{st.session_state.accuracy:.1f}%")
            if st.session_state.accuracy < 60:
                st.warning("⚠️ **Low Stability:** Results may be visually distorted due to data variance.")
            st.divider()

            all_b_labels = sorted(df_brands['Label'].tolist())
            focus_brand = st.selectbox("Highlight Column:", ["None"] + all_b_labels)
            with st.expander("Filter Base Map"):
                sel_brands = st.multiselect("Select Columns:", all_b_labels, default=all_b_labels) if not st.checkbox("All Columns", True) else all_b_labels
                all_a_labels = sorted(df_attrs['Label'].tolist())
                sel_attrs = st.multiselect("Select Rows:", all_a_labels, default=all_a_labels[:10]) if not st.checkbox("All Rows", True) else all_a_labels
            for i, layer in enumerate(passive_layer_data):
                if not layer.empty and layer['Visible'].iloc[0]:
                    with st.expander(f"Filter {layer['LayerName'].iloc[0]}"):
                        l_labels = sorted(layer['Label'].tolist())
                        if not st.checkbox("All", True, key=f"f_all_{i}"):
                            passive_layer_data[i] = layer[layer['Label'].isin(st.multiselect("Select:", l_labels, default=l_labels, key=f"f_sel_{i}"))]

        # 3. Plotting
        fig = go.Figure()
        highlight_list = []
        if focus_brand != "None":
            hero = df_brands[df_brands['Label'] == focus_brand].iloc[0]
            hx, hy = hero['x'], hero['y']
            active_attrs = df_attrs[df_attrs['Label'].isin(sel_attrs)].copy()
            active_attrs['D'] = np.sqrt((active_attrs['x']-hx)**2 + (active_attrs['y']-hy)**2)
            highlight_list += active_attrs.sort_values('D').head(5)['Label'].tolist()
            for layer in passive_layer_data:
                if not layer.empty and layer['Visible'].iloc[0]:
                    l_copy = layer.copy(); l_copy['D'] = np.sqrt((l_copy['x']-hx)**2 + (l_copy['y']-hy)**2)
                    highlight_list += l_copy.sort_values('D').head(5)['Label'].tolist()

        def get_so(lbl, base_c):
            if focus_brand == "None": return base_c, 0.9
            if lbl == focus_brand or lbl in highlight_list: return base_c, 1.0
            return '#d3d3d3', 0.25

        if show_base_cols:
            plot_b = df_brands[df_brands['Label'].isin(sel_brands)]
            c_l, o_l = zip(*[get_so(r['Label'], '#1f77b4') for _, r in plot_b.iterrows()])
            fig.add_trace(go.Scatter(x=plot_b['x'], y=plot_b['y'], mode='markers', marker=dict(size=10, color=c_l, opacity=o_l, line=dict(width=1, color='white')), text=plot_b['Label'], hoverinfo='text', name='Columns'))
            for _, r in plot_b.iterrows():
                c, o = get_so(r['Label'], '#1f77b4')
                if o > 0.4: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], ax=0, ay=-20, font=dict(color=c, size=11), arrowcolor=c)

        if show_base_rows:
            plot_a = df_attrs[df_attrs['Label'].isin(sel_attrs)]
            if enable_clustering and HAS_SKLEARN:
                for cid in sorted(plot_a['Cluster'].unique()):
                    sub = plot_a[plot_a['Cluster'] == cid]; bc = cluster_colors[cid % len(cluster_colors)]
                    c_l, o_l = zip(*[get_so(r['Label'], bc) for _, r in sub.iterrows()])
                    fig.add_trace(go.Scatter(x=sub['x'], y=sub['y'], mode='markers', marker=dict(size=7, color=c_l, opacity=o_l), text=sub['Label'], hoverinfo='text', name=f"Territory {cid+1}"))
                    for _, r in sub.iterrows():
                        c, o = get_so(r['Label'], bc)
                        if o > 0.4: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], ax=0, ay=-15, font=dict(color=c, size=11), arrowcolor=c)
            else:
                c_l, o_l = zip(*[get_so(r['Label'], '#d62728') for _, r in plot_a.iterrows()])
                fig.add_trace(go.Scatter(x=plot_a['x'], y=plot_a['y'], mode='markers', marker=dict(size=7, color=c_l, opacity=o_l), text=plot_a['Label'], hoverinfo='text', name='Base Rows'))
                for _, r in plot_a.iterrows():
                    c, o = get_so(r['Label'], '#d62728')
                    if o > 0.4: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], ax=0, ay=-15, font=dict(color=c, size=11), arrowcolor=c)

        for i, layer in enumerate(passive_layer_data):
            if not layer.empty and layer['Visible'].iloc[0]:
                if enable_clustering and 'Cluster' in layer.columns:
                    for cid in sorted(layer['Cluster'].unique()):
                        sub = layer[layer['Cluster'] == cid]; bc = cluster_colors[cid % len(cluster_colors)]
                        c_l, o_l = zip(*[get_so(r['Label'], bc) for _, r in sub.iterrows()])
                        fig.add_trace(go.Scatter(x=sub['x'], y=sub['y'], mode='markers', marker=dict(size=9, symbol=sub['Shape'].iloc[0], color=c_l, opacity=o_l, line=dict(width=1, color='white')), text=sub['Label'], hoverinfo='text', name=f"{sub['LayerName'].iloc[0]} (T{cid+1})", showlegend=False))
                        for _, r in sub.iterrows():
                            c, o = get_so(r['Label'], bc)
                            if o > 0.4: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], ax=0, ay=-15, font=dict(color=c, size=11), arrowcolor=c)
                else:
                    bc = cluster_colors[i % len(cluster_colors)]
                    c_l, o_l = zip(*[get_so(r['Label'], bc) for _, r in layer.iterrows()])
                    fig.add_trace(go.Scatter(x=layer['x'], y=layer['y'], mode='markers', marker=dict(size=9, symbol=layer['Shape'].iloc[0], color=c_l, opacity=o_l, line=dict(width=1, color='white')), text=layer['Label'], hoverinfo='text', name=layer['LayerName'].iloc[0]))
                    for _, r in layer.iterrows():
                        c, o = get_so(r['Label'], bc)
                        if o > 0.4: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], ax=0, ay=-15, font=dict(color=c, size=11), arrowcolor=c)

        fig.update_layout(title={'text': "Strategic Map", 'y':0.95, 'x':0.5, 'xanchor':'center', 'font': {'family': 'Nunito', 'size': 20}}, template="plotly_white", height=850, xaxis=dict(showgrid=False, showticklabels=False, zeroline=True), yaxis=dict(showgrid=False, showticklabels=False, zeroline=True), yaxis_scaleanchor="x", yaxis_scaleratio=1, dragmode='pan')
        st.plotly_chart(fig, use_container_width=True, config={'editable': True, 'scrollZoom': True})

        if enable_clustering and territory_report:
            st.divider()
            st.header("⚗️ Strategic Territory Briefing")
            cols = st.columns(min(3, num_clusters))
            for i, t in enumerate(territory_report):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="territory-card" style="border-left-color: {t['color']};">
                        <h3 style="color: {t['color']}; margin-top:0;">Territory {t['id']}</h3>
                        <p><b>Defining Rows:</b><br>{", ".join(t['rows'])}</p>
                        <p><b>Involved Columns:</b><br>{", ".join(t['brands']) if t['brands'] else "<i>No columns currently centered here.</i>"}</p>
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
            if "theme" in q:
                return f"**Themes:**\n* ↔️ **X-Axis:** {df_a.loc[df_a['x'].idxmin()]['Label']} to {df_a.loc[df_a['x'].idxmax()]['Label']}\n* ↕️ **Y-Axis:** {df_a.loc[df_a['y'].idxmin()]['Label']} to {df_a.loc[df_a['y'].idxmax()]['Label']}"
            for b in df_b['Label']:
                if b.lower() in q:
                    br = df_b[df_b['Label']==b].iloc[0]; df_a['D'] = np.sqrt((df_a['x']-br['x'])**2 + (df_a['y']-br['y'])**2)
                    return f"**Audit: {b}**\n✅ **Strengths:** {', '.join(df_a.sort_values('D').head(3)['Label'].tolist())}"
            return "Ask about Themes or Columns."
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        if p := st.chat_input("Ask..."):
            st.session_state.messages.append({"role": "user", "content": p}); with st.chat_message("user"): st.markdown(p)
            r = analyze_query(p); st.session_state.messages.append({"role": "assistant", "content": r}); with st.chat_message("assistant"): st.markdown(r)

# ==========================================
# TAB 3: CLEANER
# ==========================================
with tab3:
    st.header("🧹 MRI Data Cleaner")
    raw_mri = st.file_uploader("Upload Raw MRI", type=["csv", "xlsx", "xls"])
    if raw_mri:
        try:
            df_raw = pd.read_csv(raw_mri, header=None) if raw_mri.name.endswith('.csv') else pd.read_excel(raw_mri, header=None)
            metric_row_idx = next(i for i, row in df_raw.iterrows() if row.astype(str).str.contains("Weighted (000)", regex=False).any())
            brand_row = df_raw.iloc[metric_row_idx - 1]; metric_row = df_raw.iloc[metric_row_idx]; data_rows = df_raw.iloc[metric_row_idx + 1:].copy()
            cols, headers = [0], ['Attitude']
            for c in range(1, len(metric_row)):
                if "Weighted" in str(metric_row[c]):
                    brand = str(brand_row[c-1])
                    if "Study Universe" not in brand and "Total" not in brand and brand != 'nan': cols.append(c); headers.append(brand)
            df_clean = data_rows.iloc[:, cols]; df_clean.columns = headers
            df_clean['Attitude'] = df_clean['Attitude'].astype(str).str.replace('General Attitudes: ', '', regex=False)
            data_cols = df_clean.columns[1:]
            for c in data_cols: df_clean[c] = pd.to_numeric(df_clean[c].astype(str).str.replace(',', ''), errors='coerce')
            df_clean = df_clean.dropna(subset=data_cols, how='all')
            df_clean = df_clean[df_clean[data_cols].fillna(0).sum(axis=1) > 0]
            df_clean = df_clean[df_clean['Attitude'].str.len() > 3]
            df_clean = df_clean[~df_clean['Attitude'].astype(str).str.contains("Study Universe|Total|Base|Sample", case=False, regex=True)]
            st.success("Cleaned!"); st.download_button("Download CSV", df_clean.to_csv(index=False).encode('utf-8'), "Cleaned_MRI.csv", "text/csv")
        except: st.error("Could not find 'Weighted (000)' row.")
