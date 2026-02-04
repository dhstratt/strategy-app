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

# --- CUSTOM CSS: NUNITO FONT ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
        h1, h2, h3 { font-family: 'Nunito', sans-serif; font-weight: 800; }
        .stMetric { font-family: 'Nunito', sans-serif; }
        
        /* Compact Sidebar Styling */
        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
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
    # The Assassin: Kill Study Universe
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
    
    # --- SIDEBAR (STRUCTURED) ---
    with st.sidebar:
        # 1. LOAD / SAVE / IMPORT
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

        uploaded_file = st.file_uploader("Upload Core Data (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="active")
        passive_files = st.file_uploader("Upload Passive Layers", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="passive")
        
        st.divider()

        # 2. LAYER VISIBILITY (THE CONTROL PANEL)
        st.header("🎨 Layer Visibility")
        
        # Base Map Toggles
        col_base_1, col_base_2 = st.columns(2)
        show_base_cols = col_base_1.checkbox("Base Columns", True, key="viz_base_cols")
        show_base_rows = col_base_2.checkbox("Base Rows", True, key="viz_base_rows")
        
        # Passive Layer Configs
        passive_configs = []
        if passive_files:
            st.subheader("Passive Layers")
            for p_file in passive_files:
                with st.expander(f"⚙️ {p_file.name}", expanded=True):
                    # Configs for this file
                    p_name = st.text_input("Label", p_file.name, key=f"name_{p_file.name}")
                    p_show = st.checkbox("Show Layer", True, key=f"show_{p_file.name}")
                    p_mode = st.radio("Map As:", ["Auto", "Rows (Stars)", "Columns (Diamonds)"], key=f"mode_{p_file.name}")
                    
                    passive_configs.append({
                        "file": p_file, "name": p_name, "show": p_show, "mode": p_mode
                    })
        
        st.divider()

        # 3. ANALYSIS (CLUSTERING)
        st.header("⚗️ Analysis")
        enable_clustering = st.checkbox("Enable Territory Discovery", False)
        num_clusters = 4
        if enable_clustering:
            if HAS_SKLEARN:
                num_clusters = st.slider("Number of Territories", 2, 8, 4)
            else:
                st.error("⚠️ Library 'scikit-learn' missing. Please add to requirements.txt")
        
        st.divider()

        # 4. FILTERS & SPOTLIGHT
        st.header("🔍 Filter & Highlight")
        
        # Will be populated after data processing...
        placeholder_filters = st.empty()


    # --- DATA PROCESSING LOGIC ---
    if uploaded_file is not None:
        try:
            # Process Core
            df_raw = load_file(uploaded_file)
            df_math = clean_df(df_raw)
            # Remove all-zero rows/cols from map
            df_math = df_math.loc[(df_math != 0).any(axis=1)] 
            df_math = df_math.loc[:, (df_math != 0).any(axis=0)]
            
            if not df_math.empty:
                # Math
                N = df_math.values
                P = N / N.sum()
                r = P.sum(axis=1)
                c = P.sum(axis=0)
                E = np.outer(r, c)
                E[E < 1e-9] = 1e-9
                R = (P - E) / np.sqrt(E)
                U, s, Vh = np.linalg.svd(R, full_matrices=False)
                
                inertia = s**2
                map_accuracy = (np.sum(inertia[:2]) / np.sum(inertia)) * 100
                
                row_coords = (U * s) / np.sqrt(r[:, np.newaxis])
                col_coords = (Vh.T * s) / np.sqrt(c[:, np.newaxis])
                
                df_brands = pd.DataFrame(col_coords[:, :2], columns=['x', 'y'])
                df_brands['Label'] = df_math.columns
                df_brands['Type'] = 'Column'
                df_brands['Distinctiveness'] = np.sqrt(df_brands['x']**2 + df_brands['y']**2)
                
                df_attrs = pd.DataFrame(row_coords[:, :2], columns=['x', 'y'])
                df_attrs['Label'] = df_math.index
                df_attrs['Type'] = 'Row'
                df_attrs['Distinctiveness'] = np.sqrt(df_attrs['x']**2 + df_attrs['y']**2)

                st.session_state.processed_data = True
                st.session_state.df_brands = df_brands
                st.session_state.df_attrs = df_attrs
                st.session_state.accuracy = map_accuracy

                # Process Passive
                passive_layer_data = []
                for cfg in passive_configs:
                    try:
                        p_df = load_file(cfg["file"])
                        p_clean = clean_df(p_df)
                        
                        common_brands = list(set(p_clean.columns) & set(df_math.columns))
                        common_attrs = list(set(p_clean.index) & set(df_math.index))
                        
                        # Determine Mode
                        is_rows = False
                        if cfg["mode"] == "Rows (Stars)": is_rows = True
                        elif cfg["mode"] == "Columns (Diamonds)": is_rows = False
                        else: is_rows = True if len(common_brands) > len(common_attrs) else False
                        
                        # --- DE-DUPLICATION ---
                        if is_rows:
                            unique_rows = [r for r in p_clean.index if r not in df_math.index]
                            p_clean = p_clean.loc[unique_rows]
                            if p_clean.empty: continue
                            
                            if len(common_brands) > 0:
                                p_aligned = p_clean[common_brands].reindex(columns=df_math.columns).fillna(0)
                                p_prof = p_aligned.div(p_aligned.sum(axis=1).replace(0,1), axis=0)
                                proj = p_prof.values @ col_coords[:, :2] / s[:2]
                                res = pd.DataFrame(proj, columns=['x', 'y'])
                                res['Label'] = p_aligned.index
                                res['Shape'] = 'star'
                                res['LayerName'] = cfg["name"]
                                res['Visible'] = cfg["show"]
                                passive_layer_data.append(res)
                        else:
                            unique_cols = [c for c in p_clean.columns if c not in df_math.columns]
                            p_clean = p_clean[unique_cols]
                            if p_clean.empty: continue
                            
                            if len(common_attrs) > 0:
                                p_aligned = p_clean.reindex(df_math.index).fillna(0)
                                p_prof = p_aligned.div(p_aligned.sum(axis=0).replace(0,1), axis=1)
                                proj = p_prof.T.values @ row_coords[:, :2] / s[:2]
                                res = pd.DataFrame(proj, columns=['x', 'y'])
                                res['Label'] = p_aligned.columns
                                res['Shape'] = 'diamond'
                                res['LayerName'] = cfg["name"]
                                res['Visible'] = cfg["show"]
                                passive_layer_data.append(res)
                    except: pass
                st.session_state.passive_data = passive_layer_data

        except Exception as e: st.error(f"Processing Error: {e}")

    # --- RENDER MAP ---
    if st.session_state.processed_data:
        df_brands = st.session_state.df_brands
        df_attrs = st.session_state.df_attrs
        passive_layer_data = st.session_state.passive_data
        
        # --- CLUSTERING LOGIC ---
        # We need to persist the model to apply it to passive layers
        kmeans_model = None
        cluster_colors = px.colors.qualitative.Bold
        
        if enable_clustering and HAS_SKLEARN:
            try:
                # 1. Fit on Base Rows (The "Ground Truth")
                kmeans_model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                df_attrs['Cluster'] = kmeans_model.fit_predict(df_attrs[['x', 'y']])
                df_attrs['Cluster_Label'] = "Territory " + (df_attrs['Cluster'] + 1).astype(str)
                
                # 2. Predict on Passive Layers (if they exist)
                for i, layer in enumerate(passive_layer_data):
                    if not layer.empty:
                        # Predict which territory these new points fall into
                        layer['Cluster'] = kmeans_model.predict(layer[['x', 'y']])
                        passive_layer_data[i] = layer # Update the list
                        
            except Exception as e:
                # Fallback if clustering fails
                df_attrs['Cluster_Label'] = "Base Rows"
        else:
             df_attrs['Cluster_Label'] = "Base Rows"

        # --- POPULATE FILTERS ---
        with placeholder_filters.container():
            all_b_labels = sorted(df_brands['Label'].tolist())
            focus_brand = st.selectbox("Highlight Column:", ["None"] + all_b_labels)
            
            with st.expander("Filter Base Map", expanded=False):
                if not st.checkbox("All Columns", True, key="f_all_cols"):
                    sel_brands = st.multiselect("Select Columns:", all_b_labels, default=all_b_labels, key="f_sel_cols")
                else: sel_brands = all_b_labels
                
                all_a_labels = sorted(df_attrs['Label'].tolist())
                if not st.checkbox("All Rows", True, key="f_all_rows"):
                    sel_attrs = st.multiselect("Select Rows:", all_a_labels, default=all_a_labels[:10], key="f_sel_rows")
                else: sel_attrs = all_a_labels
            
            for i, layer in enumerate(passive_layer_data):
                if not layer.empty and layer['Visible'].iloc[0]:
                    with st.expander(f"Filter {layer['LayerName'].iloc[0]}", expanded=False):
                        l_labels = sorted(layer['Label'].tolist())
                        if not st.checkbox("All", True, key=f"f_all_pass_{i}"):
                            sel_l = st.multiselect("Select:", l_labels, default=l_labels, key=f"f_sel_pass_{i}")
                            passive_layer_data[i] = layer[layer['Label'].isin(sel_l)]

        # --- PREPARE PLOT ---
        fig = go.Figure()
        
        # Spotlight Logic
        highlight_list = []
        if focus_brand != "None":
            hero = df_brands[df_brands['Label'] == focus_brand]
            if not hero.empty:
                hx, hy = hero.iloc[0]['x'], hero.iloc[0]['y']
                
                # Base Rows
                if show_base_rows:
                    active_attrs = df_attrs[df_attrs['Label'].isin(sel_attrs)].copy()
                    active_attrs['D'] = np.sqrt((active_attrs['x']-hx)**2 + (active_attrs['y']-hy)**2)
                    highlight_list += active_attrs.sort_values('D').head(5)['Label'].tolist()
                
                # Passive Items
                for layer in passive_layer_data:
                    if not layer.empty and layer['Visible'].iloc[0]:
                        l_copy = layer.copy()
                        l_copy['D'] = np.sqrt((l_copy['x']-hx)**2 + (l_copy['y']-hy)**2)
                        highlight_list += l_copy.sort_values('D').head(5)['Label'].tolist()

        def get_color_opacity(label, base_color):
            if focus_brand == "None":
                return base_color, 0.9
            if label == focus_brand:
                return base_color, 1.0
            if label in highlight_list:
                return base_color, 1.0
            return '#d3d3d3', 0.25

        # 1. BASE COLUMNS
        if show_base_cols:
            plot_b = df_brands[df_brands['Label'].isin(sel_brands)]
            c_list, o_list = [], []
            for _, r in plot_b.iterrows():
                # Columns usually stay blue unless we want them clustered too.
                # Standard convention: Brands are entities, Rows are the landscape.
                # So we keep brands Blue (or highlight color).
                c, o = get_color_opacity(r['Label'], '#1f77b4') 
                c_list.append(c); o_list.append(o)
            
            fig.add_trace(go.Scatter(
                x=plot_b['x'], y=plot_b['y'], mode='markers',
                marker=dict(size=10, color=c_list, opacity=o_list, line=dict(width=1, color='white')),
                text=plot_b['Label'], hoverinfo='text', name='Base Columns'
            ))
            for _, r in plot_b.iterrows():
                c, o = get_color_opacity(r['Label'], '#1f77b4')
                if o > 0.4: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], ax=0, ay=-20, font=dict(color=c, size=11), arrowcolor=c)

        # 2. BASE ROWS
        if show_base_rows:
            plot_a = df_attrs[df_attrs['Label'].isin(sel_attrs)]
            
            if enable_clustering and HAS_SKLEARN:
                # Render by Cluster
                clusters = sorted(plot_a['Cluster'].unique())
                for c_id in clusters:
                    sub_df = plot_a[plot_a['Cluster'] == c_id]
                    base_c = cluster_colors[c_id % len(cluster_colors)]
                    
                    c_list, o_list = [], []
                    for _, r in sub_df.iterrows():
                        c, o = get_color_opacity(r['Label'], base_c)
                        c_list.append(c); o_list.append(o)
                    
                    fig.add_trace(go.Scatter(
                        x=sub_df['x'], y=sub_df['y'], mode='markers',
                        marker=dict(size=7, color=c_list, opacity=o_list),
                        text=sub_df['Label'], hoverinfo='text', name=f"Territory {c_id+1}"
                    ))
                    # Annotations loop for this cluster
                    for _, r in sub_df.iterrows():
                         c, o = get_color_opacity(r['Label'], base_c)
                         if o > 0.4: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], ax=0, ay=-15, font=dict(color=c, size=11), arrowcolor=c)

            else:
                # Standard Red
                c_list, o_list = [], []
                for _, r in plot_a.iterrows():
                    c, o = get_color_opacity(r['Label'], '#d62728')
                    c_list.append(c); o_list.append(o)
                fig.add_trace(go.Scatter(
                    x=plot_a['x'], y=plot_a['y'], mode='markers',
                    marker=dict(size=7, color=c_list, opacity=o_list),
                    text=plot_a['Label'], hoverinfo='text', name='Base Rows'
                ))
                for _, r in plot_a.iterrows():
                    c, o = get_color_opacity(r['Label'], '#d62728')
                    if o > 0.4: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], ax=0, ay=-15, font=dict(color=c, size=11), arrowcolor=c)

        # 3. PASSIVE LAYERS
        pass_colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b'] # Fallback
        
        for i, layer in enumerate(passive_layer_data):
            if not layer.empty and layer['Visible'].iloc[0]:
                
                # If Clustering ON, split this layer into cluster groups
                if enable_clustering and HAS_SKLEARN and 'Cluster' in layer.columns:
                    
                    # We iterate through clusters found in THIS layer
                    layer_clusters = sorted(layer['Cluster'].unique())
                    for c_id in layer_clusters:
                        sub_layer = layer[layer['Cluster'] == c_id]
                        # Use same color as Base Rows for this cluster
                        base_c = cluster_colors[c_id % len(cluster_colors)]
                        
                        c_list, o_list = [], []
                        for _, r in sub_layer.iterrows():
                            c, o = get_color_opacity(r['Label'], base_c)
                            c_list.append(c); o_list.append(o)

                        # Name: "Layer Name (T1)"
                        fig.add_trace(go.Scatter(
                            x=sub_layer['x'], y=sub_layer['y'], mode='markers',
                            marker=dict(size=9, symbol=sub_layer['Shape'].iloc[0], color=c_list, opacity=o_list, line=dict(width=1, color='white')),
                            text=sub_layer['Label'], hoverinfo='text', 
                            name=f"{sub_layer['LayerName'].iloc[0]} (T{c_id+1})",
                            showlegend=False # Cleaner to hide sub-legends, or keep if user wants detail
                        ))
                        
                        for _, r in sub_layer.iterrows():
                            c, o = get_color_opacity(r['Label'], base_c)
                            if o > 0.4: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], ax=0, ay=-15, font=dict(color=c, size=11), arrowcolor=c)

                else:
                    # Standard Layer Color
                    pc = pass_colors[i % len(pass_colors)]
                    c_list, o_list = [], []
                    for _, r in layer.iterrows():
                        c, o = get_color_opacity(r['Label'], pc)
                        c_list.append(c); o_list.append(o)
                    
                    fig.add_trace(go.Scatter(
                        x=layer['x'], y=layer['y'], mode='markers',
                        marker=dict(size=9, symbol=layer['Shape'].iloc[0], color=c_list, opacity=o_list, line=dict(width=1, color='white')),
                        text=layer['Label'], hoverinfo='text', name=layer['LayerName'].iloc[0]
                    ))
                    for _, r in layer.iterrows():
                        c, o = get_color_opacity(r['Label'], pc)
                        if o > 0.4: fig.add_annotation(x=r['x'], y=r['y'], text=r['Label'], ax=0, ay=-15, font=dict(color=c, size=11), arrowcolor=c)

        # Layout
        fig.update_layout(
            title={'text': "Strategic Map", 'y':0.95, 'x':0.5, 'xanchor':'center', 'font': {'family': 'Nunito', 'size': 20}},
            template="plotly_white", height=850,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=True),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=True),
            yaxis_scaleanchor="x", yaxis_scaleratio=1, dragmode='pan'
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'editable': True, 'scrollZoom': True, 'displayModeBar': True})

# ==========================================
# TAB 2: AI CHAT
# ==========================================
with tab2:
    st.header("💬 AI Landscape Chat")
    if not st.session_state.processed_data: st.warning("👈 Upload data first.")
    else:
        def analyze_query(query):
            q, df_b, df_a = query.lower(), st.session_state.df_brands, st.session_state.df_attrs
            if "theme" in q or "axis" in q:
                xm, xn = df_a.loc[df_a['x'].idxmax()]['Label'], df_a.loc[df_a['x'].idxmin()]['Label']
                ym, yn = df_a.loc[df_a['y'].idxmax()]['Label'], df_a.loc[df_a['y'].idxmin()]['Label']
                return f"**Themes:**\n* ↔️ **X-Axis:** {xn} to {xm}\n* ↕️ **Y-Axis:** {yn} to {ym}"
            if "white space" in q:
                df_a['D'] = df_a.apply(lambda r: np.min(np.sqrt((df_b['x']-r['x'])**2 + (df_b['y']-r['y'])**2)), axis=1)
                ws = df_a.sort_values('D', ascending=False).head(3)
                return "**White Space:**\n" + "\n".join([f"* {r['Label']}" for _, r in ws.iterrows()])
            for b in df_b['Label']:
                if b.lower() in q:
                    br = df_b[df_b['Label']==b].iloc[0]
                    df_a['D'] = np.sqrt((df_a['x']-br['x'])**2 + (df_a['y']-br['y'])**2)
                    df_a['O'] = np.sqrt((df_a['x']-(-br['x']))**2 + (df_a['y']-(-br['y']))**2)
                    df_b['D'] = np.sqrt((df_b['x']-br['x'])**2 + (df_b['y']-br['y'])**2)
                    s = df_a.sort_values('D').head(3)['Label'].tolist()
                    w = df_a.sort_values('O').head(3)['Label'].tolist()
                    c = df_b[df_b['Label']!=b].sort_values('D').head(3)['Label'].tolist()
                    return f"**Audit: {b}**\n✅ **Strengths:** {', '.join(s)}\n❌ **Weaknesses:** {', '.join(w)}\n⚔️ **Competitors:** {', '.join(c)}"
            return "Ask about **Themes**, **White Space**, or **Audit [Column]**."

        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        if p := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": p})
            with st.chat_message("user"): st.markdown(p)
            r = analyze_query(p)
            with st.chat_message("assistant"): st.markdown(r)
            st.session_state.messages.append({"role": "assistant", "content": r})

# ==========================================
# TAB 3: CLEANER
# ==========================================
with tab3:
    st.header("🧹 MRI Data Cleaner")
    raw_mri = st.file_uploader("Upload Raw MRI", type=["csv", "xlsx", "xls"])
    if raw_mri:
        try:
            if raw_mri.name.endswith('.csv'): df_raw = pd.read_csv(raw_mri, header=None)
            else: df_raw = pd.read_excel(raw_mri, header=None)
            
            metric_row_idx = -1
            for i, row in df_raw.iterrows():
                if row.astype(str).str.contains("Weighted (000)", regex=False).any():
                    metric_row_idx = i; break
            
            if metric_row_idx != -1:
                brand_row = df_raw.iloc[metric_row_idx - 1]
                metric_row = df_raw.iloc[metric_row_idx]
                data_rows = df_raw.iloc[metric_row_idx + 1:].copy()
                cols, headers = [0], ['Attitude']
                for c in range(1, len(metric_row)):
                    if "Weighted" in str(metric_row[c]):
                        brand = str(brand_row[c-1])
                        if "Study Universe" not in brand and "Total" not in brand and brand != 'nan':
                            cols.append(c); headers.append(brand)
                
                df_clean = data_rows.iloc[:, cols]; df_clean.columns = headers
                df_clean['Attitude'] = df_clean['Attitude'].astype(str).str.replace('General Attitudes: ', '', regex=False)
                
                # --- JUNK REMOVAL ---
                data_cols = df_clean.columns[1:]
                for c in data_cols:
                    df_clean[c] = pd.to_numeric(df_clean[c].astype(str).str.replace(',', ''), errors='coerce')
                
                df_clean = df_clean.dropna(subset=data_cols, how='all')
                df_clean = df_clean[df_clean[data_cols].fillna(0).sum(axis=1) > 0]

                # Filter Tags
                df_clean = df_clean[df_clean['Attitude'].str.len() > 3]
                df_clean = df_clean[~df_clean['Attitude'].astype(str).str.contains("Study Universe|Total|Base|Sample", case=False, regex=True)]
                
                st.success("Cleaned!")
                st.download_button("Download CSV", df_clean.to_csv(index=False).encode('utf-8'), "Cleaned_MRI.csv", "text/csv")
            else: st.error("Could not find 'Weighted (000)' row.")
        except Exception as e: st.error(f"Error: {e}")
