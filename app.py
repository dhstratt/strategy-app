import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import collections
import io
import pickle

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="The Consumer Landscape")

# --- CUSTOM CSS: NUNITO FONT ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
        h1, h2, h3 { font-family: 'Nunito', sans-serif; font-weight: 800; }
        .stMetric { font-family: 'Nunito', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = False
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am your Landscape Guide. Ask me about **Themes**, **White Space**, or specific **Brands**."}
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
# TAB 1: THE STRATEGY ENGINE
# ==========================================
with tab1:
    st.title("🗺️ The Consumer Landscape")
    col_nav, col_main = st.columns([1, 4])
    
    with st.sidebar:
        st.header("💾 Project Manager")
        with st.expander("📂 Load Project", expanded=False):
            uploaded_project = st.file_uploader("Upload .use file", type=["use"], key="loader")
            if uploaded_project is not None:
                try:
                    data = pickle.load(uploaded_project)
                    st.session_state.df_brands = data['df_brands']
                    st.session_state.df_attrs = data['df_attrs']
                    st.session_state.passive_data = data['passive_data']
                    st.session_state.accuracy = data['accuracy']
                    st.session_state.processed_data = True
                    st.success("Project Loaded!")
                    st.rerun()
                except: st.error("Invalid Project File")

        with st.expander("💾 Save Project", expanded=True):
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
                st.download_button("Download Project 📥", buffer, f"{proj_name}.use", "application/octet-stream")
            else: st.info("Upload data to save.")

        st.divider()
        st.header("📂 Data Import")
        uploaded_file = st.file_uploader("1. Upload Core Data", type=["csv", "xlsx", "xls"], key="active")
        passive_files = st.file_uploader("2. Upload Passive Layers", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="passive")
        st.divider()

    # --- PROCESS DATA ---
    if uploaded_file is not None:
        try:
            df_active_raw = load_file(uploaded_file)
            df_math = clean_df(df_active_raw)
            df_math = df_math.loc[(df_math != 0).any(axis=1)] 
            df_math = df_math.loc[:, (df_math != 0).any(axis=0)]
            
            if not df_math.empty:
                # SVD
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
                df_brands['Type'] = 'Brand'
                df_brands['Distinctiveness'] = np.sqrt(df_brands['x']**2 + df_brands['y']**2)
                
                df_attrs = pd.DataFrame(row_coords[:, :2], columns=['x', 'y'])
                df_attrs['Label'] = df_math.index
                df_attrs['Type'] = 'Attribute'
                df_attrs['Distinctiveness'] = np.sqrt(df_attrs['x']**2 + df_attrs['y']**2)

                st.session_state.processed_data = True
                st.session_state.df_brands = df_brands
                st.session_state.df_attrs = df_attrs
                st.session_state.accuracy = map_accuracy

                # Passive
                passive_layer_data = []
                if passive_files:
                    for p_file in passive_files:
                        try:
                            p_raw = load_file(p_file)
                            p_clean = clean_df(p_raw)
                            common_brands = list(set(p_clean.columns) & set(df_math.columns))
                            common_attrs = list(set(p_clean.index) & set(df_math.index))
                            
                            if len(common_brands) > len(common_attrs): # Attributes
                                p_aligned = p_clean[common_brands].reindex(columns=df_math.columns).fillna(0)
                                p_prof = p_aligned.div(p_aligned.sum(axis=1).replace(0,1), axis=0)
                                proj = p_prof.values @ col_coords[:, :2] / s[:2]
                                res = pd.DataFrame(proj, columns=['x', 'y'])
                                res['Label'] = p_aligned.index; res['Shape'] = 'star'
                            else: # Brands
                                p_aligned = p_clean.reindex(df_math.index).fillna(0)
                                p_prof = p_aligned.div(p_aligned.sum(axis=0).replace(0,1), axis=1)
                                proj = p_prof.T.values @ row_coords[:, :2] / s[:2]
                                res = pd.DataFrame(proj, columns=['x', 'y'])
                                res['Label'] = p_aligned.columns; res['Shape'] = 'diamond'
                            
                            res['LayerName'] = p_file.name
                            res['Distinctiveness'] = np.sqrt(res['x']**2 + res['y']**2)
                            passive_layer_data.append(res)
                        except: pass
                st.session_state.passive_data = passive_layer_data
        except Exception as e: st.error(f"Error: {e}")

    # --- RENDER MAP ---
    if st.session_state.processed_data:
        df_brands = st.session_state.df_brands
        df_attrs = st.session_state.df_attrs
        passive_layer_data = st.session_state.passive_data
        
        # 1. CALCULATE BOUNDS FOR AUTO-ZOOM
        # Combine all visible coordinates to find the max extent
        all_x = pd.concat([df_brands['x'], df_attrs['x']])
        all_y = pd.concat([df_brands['y'], df_attrs['y']])
        
        # Check passive layers too
        for layer in passive_layer_data:
            all_x = pd.concat([all_x, layer['x']])
            all_y = pd.concat([all_y, layer['y']])
            
        max_bound = max(all_x.abs().max(), all_y.abs().max()) * 1.1 # Add 10% padding
        
        # CONTROLS
        with st.sidebar:
            st.header("🎯 Map Controls")
            st.metric("Map Stability", f"{st.session_state.accuracy:.1f}%")
            if st.session_state.accuracy < 60: st.error("⚠️ **Unstable Map:** Accuracy < 60%.")

            st.markdown("---")
            st.subheader("🔦 Brand Spotlight")
            all_b = sorted(df_brands['Label'].tolist())
            focus_brand = st.selectbox("Highlight:", ["None"] + all_b)
            st.markdown("---")

            # Core Filters
            with st.expander("🔹 Core Brands", expanded=False):
                if not st.checkbox("Show All Brands", value=True):
                    sel_brands = st.multiselect("Filter:", all_b, default=all_b)
                else: sel_brands = df_brands['Label'].tolist()

            with st.expander("🔹 Core Statements", expanded=False):
                if not st.checkbox("Show All Statements", value=False):
                    all_a = sorted(df_attrs['Label'].tolist())
                    top_15 = df_attrs.sort_values('Distinctiveness', ascending=False).head(15)['Label'].tolist()
                    sel_attrs = st.multiselect("Filter:", all_a, default=top_15)
                else: sel_attrs = df_attrs['Label'].tolist()

            # Passive Filters
            sel_passive_data = []
            if passive_layer_data:
                st.subheader("🔸 Passive Layers")
                for layer in passive_layer_data:
                    lname = layer['LayerName'].iloc[0]
                    if st.checkbox(f"👁️ {lname}", value=True, key=lname):
                        with st.expander(f"Filter {lname}", expanded=False):
                            if not st.checkbox(f"Select All", value=True, key=f"all_{lname}"):
                                all_l = sorted(layer['Label'].tolist())
                                sel_l = st.multiselect("Filter Items:", all_l, default=all_l, key=f"mul_{lname}")
                            else: sel_l = layer['Label'].tolist()
                            if sel_l: sel_passive_data.append(layer[layer['Label'].isin(sel_l)])

        # PLOT LOGIC
        plot_brands = df_brands[df_brands['Label'].isin(sel_brands)]
        plot_attrs = df_attrs[df_attrs['Label'].isin(sel_attrs)]
        
        hero_related = []
        if focus_brand != "None":
            hero = df_brands[df_brands['Label'] == focus_brand]
            if not hero.empty:
                hx, hy = hero.iloc[0]['x'], hero.iloc[0]['y']
                plot_attrs = plot_attrs.copy()
                plot_attrs['Dist'] = np.sqrt((plot_attrs['x'] - hx)**2 + (plot_attrs['y'] - hy)**2)
                hero_related = plot_attrs.sort_values('Dist').head(5)['Label'].tolist()

        fig = go.Figure()

        def get_style(lbl, is_brand=False):
            c = '#1f77b4' if is_brand else '#d62728'
            op = 1.0 if is_brand else 0.7
            if focus_brand != "None":
                if is_brand: return (c, 1.0) if lbl == focus_brand else ('#d3d3d3', 0.2)
                else: return (c, 1.0) if lbl in hero_related else ('#d3d3d3', 0.2)
            return c, op

        # Brands
        bc, bo = [], []
        for _, r in plot_brands.iterrows():
            c, o = get_style(r['Label'], True)
            bc.append(c); bo.append(o)
        fig.add_trace(go.Scatter(x=plot_brands['x'], y=plot_brands['y'], mode='markers', marker=dict(size=10, color=bc, opacity=bo, line=dict(width=1, color='white')), hovertext=plot_brands['Label'], name='Brands'))

        # Attributes
        ac, ao = [], []
        for _, r in plot_attrs.iterrows():
            c, o = get_style(r['Label'], False)
            ac.append(c); ao.append(o)
        fig.add_trace(go.Scatter(x=plot_attrs['x'], y=plot_attrs['y'], mode='markers', marker=dict(size=7, color=ac, opacity=ao), hovertext=plot_attrs['Label'], name='Statements'))

        # Passive
        pass_colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
        for i, layer in enumerate(sel_passive_data):
            pc = pass_colors[i % len(pass_colors)]
            lc = [pc if focus_brand == "None" else '#d3d3d3' for _ in range(len(layer))]
            lo = [0.9 if focus_brand == "None" else 0.2 for _ in range(len(layer))]
            fig.add_trace(go.Scatter(x=layer['x'], y=layer['y'], mode='markers', marker=dict(size=9, symbol=layer['Shape'].iloc[0], color=lc, opacity=lo, line=dict(width=1, color='white')), hovertext=layer['Label'], name=layer['LayerName'].iloc[0]))

        # Annotations (Clean: No Background Box)
        anns = []
        for _, r in plot_brands.iterrows():
            c, o = get_style(r['Label'], True)
            if o > 0.3: anns.append(dict(x=r['x'], y=r['y'], text=r['Label'], ax=0, ay=-20, font=dict(size=11, color=c, family="Nunito"), arrowcolor=c))
        
        for _, r in plot_attrs.iterrows():
            c, o = get_style(r['Label'], False)
            if o > 0.3: anns.append(dict(x=r['x'], y=r['y'], text=r['Label'], ax=0, ay=-15, font=dict(size=11, color=c, family="Nunito"), arrowcolor=c))

        for i, layer in enumerate(sel_passive_data):
            if focus_brand == "None":
                pc = pass_colors[i % len(pass_colors)]
                for _, r in layer.iterrows():
                    anns.append(dict(x=r['x'], y=r['y'], text=r['Label'], ax=0, ay=-15, font=dict(size=11, color=pc, family="Nunito"), arrowcolor=pc))

        # Layout (Auto-Zoom Fix)
        fig.update_layout(
            annotations=anns,
            title={'text': "Strategic Map", 'y':0.95, 'x':0.5, 'xanchor':'center', 'font': {'family': 'Nunito', 'size': 20}},
            template="plotly_white", height=850,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=True, range=[-max_bound, max_bound]),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=True, range=[-max_bound, max_bound]),
            yaxis_scaleanchor="x", yaxis_scaleratio=1,
            dragmode='pan'
        )
        
        # Quadrants (Dynamic Size)
        fig.add_shape(type="rect", x0=0, y0=0, x1=max_bound, y1=max_bound, fillcolor="blue", opacity=0.03, layer="below", line_width=0)
        fig.add_shape(type="rect", x0=-max_bound, y0=-max_bound, x1=0, y1=0, fillcolor="blue", opacity=0.03, layer="below", line_width=0)

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
            return "Ask about **Themes**, **White Space**, or **Audit [Brand]**."

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
                if row.astype(str).str.contains("Weighted (000)", regex=False).any(): metric_row_idx = i; break
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
                df_clean = df_clean[df_clean['Attitude'].str.len() > 3]
                df_clean = df_clean[~df_clean['Attitude'].astype(str).str.contains("Study Universe|Total|Base|Sample", case=False, regex=True)]
                st.success("Cleaned!")
                st.download_button("Download CSV", df_clean.to_csv(index=False).encode('utf-8'), "Cleaned_MRI.csv", "text/csv")
            else: st.error("Could not find 'Weighted (000)' row.")
        except Exception as e: st.error(f"Error: {e}")
