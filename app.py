import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import collections
import io

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Universal Strategy Engine")

# --- SESSION STATE INITIALIZATION ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = False

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🧠 The Strategy Engine", "📝 Interpretation", "🧹 MRI Data Cleaner"])

# ==========================================
# TAB 1: THE STRATEGY ENGINE
# ==========================================
with tab1:
    st.title("🧠 The Strategy Engine")
    
    col_nav, col_main = st.columns([1, 4])
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("📂 Data Manager")
        uploaded_file = st.file_uploader("1. Upload Core Data", type=["csv", "xlsx", "xls"], key="active")
        passive_files = st.file_uploader("2. Upload Passive Layers", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="passive")
        st.divider()

    def load_file(file):
        if file.name.endswith('.csv'): return pd.read_csv(file)
        else: return pd.read_excel(file)

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

    if uploaded_file is not None:
        try:
            # --- PROCESS DATA ---
            df_active_raw = load_file(uploaded_file)
            df_math = clean_df(df_active_raw)
            df_math = df_math.loc[(df_math != 0).any(axis=1)] 
            df_math = df_math.loc[:, (df_math != 0).any(axis=0)]
            
            # SVD
            N = df_math.values
            P = N / N.sum()
            r = P.sum(axis=1)
            c = P.sum(axis=0)
            E = np.outer(r, c)
            E[E < 1e-9] = 1e-9
            R = (P - E) / np.sqrt(E)
            U, s, Vh = np.linalg.svd(R, full_matrices=False)
            
            # Metric
            inertia = s**2
            map_accuracy = (np.sum(inertia[:2]) / np.sum(inertia)) * 100
            
            # Coordinates
            row_coords = (U * s) / np.sqrt(r[:, np.newaxis])
            col_coords = (Vh.T * s) / np.sqrt(c[:, np.newaxis])
            
            # DataFrames
            df_brands = pd.DataFrame(col_coords[:, :2], columns=['x', 'y'])
            df_brands['Label'] = df_math.columns
            df_brands['Type'] = 'Brand (Core)'
            df_brands['Distinctiveness'] = np.sqrt(df_brands['x']**2 + df_brands['y']**2)
            
            df_attrs = pd.DataFrame(row_coords[:, :2], columns=['x', 'y'])
            df_attrs['Label'] = df_math.index
            df_attrs['Type'] = 'Attribute'
            df_attrs['Distinctiveness'] = np.sqrt(df_attrs['x']**2 + df_attrs['y']**2)

            # Passive
            passive_layer_data = []
            if passive_files:
                for p_file in passive_files:
                    try:
                        p_raw = load_file(p_file)
                        p_clean = clean_df(p_raw)
                        common_brands = list(set(p_clean.columns) & set(df_math.columns))
                        common_attrs = list(set(p_clean.index) & set(df_math.index))
                        
                        if len(common_brands) > len(common_attrs):
                            p_aligned = p_clean[common_brands].reindex(columns=df_math.columns).fillna(0)
                            p_profiles = p_aligned.div(p_aligned.sum(axis=1).replace(0,1), axis=0)
                            proj = p_profiles.values @ col_coords[:, :2] / s[:2]
                            res = pd.DataFrame(proj, columns=['x', 'y'])
                            res['Label'] = p_aligned.index
                            res['Shape'] = 'star'
                        else:
                            p_aligned = p_clean.reindex(df_math.index).fillna(0)
                            p_profiles = p_aligned.div(p_aligned.sum(axis=0).replace(0,1), axis=1)
                            proj = p_profiles.T.values @ row_coords[:, :2] / s[:2]
                            res = pd.DataFrame(proj, columns=['x', 'y'])
                            res['Label'] = p_aligned.columns
                            res['Shape'] = 'diamond'
                        
                        res['LayerName'] = p_file.name
                        res['Distinctiveness'] = np.sqrt(res['x']**2 + res['y']**2)
                        passive_layer_data.append(res)
                    except: pass
            
            # --- SAVE STATE FOR TAB 2 ---
            st.session_state.processed_data = True
            st.session_state.df_brands = df_brands
            st.session_state.df_attrs = df_attrs
            st.session_state.passive_data = passive_layer_data
            st.session_state.accuracy = map_accuracy

            # --- CONTROLS ---
            with st.sidebar:
                st.header("🎯 Map Controls")
                st.metric("Map Stability", f"{map_accuracy:.1f}%")
                
                # --- SPOTLIGHT ---
                st.markdown("---")
                st.subheader("🔦 Brand Spotlight")
                all_brand_labels = sorted(df_brands['Label'].tolist())
                focus_brand = st.selectbox("Highlight a Brand Story:", ["None"] + all_brand_labels)
                st.markdown("---")

                with st.expander("🔹 Core Brands", expanded=False):
                    if not st.checkbox("Show All Brands", value=True):
                        sel_brands = st.multiselect("Filter:", all_brand_labels, default=all_brand_labels)
                    else: sel_brands = df_brands['Label'].tolist()

                with st.expander("🔹 Core Statements", expanded=False):
                    if not st.checkbox("Show All Statements", value=False):
                        all_a = sorted(df_attrs['Label'].tolist())
                        top_15 = df_attrs.sort_values('Distinctiveness', ascending=False).head(15)['Label'].tolist()
                        sel_attrs = st.multiselect("Filter:", all_a, default=top_15)
                    else: sel_attrs = df_attrs['Label'].tolist()

                sel_passive_data = []
                if passive_layer_data:
                    st.subheader("🔸 Passive Layers")
                    for layer in passive_layer_data:
                        lname = layer['LayerName'].iloc[0]
                        show_layer = st.checkbox(f"👁️ {lname}", value=True, key=f"vis_{lname}")
                        if show_layer:
                            with st.expander(f"Filter {lname}", expanded=False):
                                if not st.checkbox(f"Select All", value=True, key=f"all_{lname}"):
                                    all_l = sorted(layer['Label'].tolist())
                                    sel_l = st.multiselect("Filter Items:", all_l, default=all_l, key=f"mul_{lname}")
                                else: sel_l = layer['Label'].tolist()
                                if sel_l: sel_passive_data.append(layer[layer['Label'].isin(sel_l)])

            # --- SPOTLIGHT LOGIC ---
            plot_brands = df_brands[df_brands['Label'].isin(sel_brands)]
            plot_attrs = df_attrs[df_attrs['Label'].isin(sel_attrs)]
            
            hero_related_attrs = []
            if focus_brand != "None":
                hero_row = df_brands[df_brands['Label'] == focus_brand]
                if not hero_row.empty:
                    hero_x, hero_y = hero_row.iloc[0]['x'], hero_row.iloc[0]['y']
                    plot_attrs = plot_attrs.copy()
                    plot_attrs['DistToHero'] = np.sqrt((plot_attrs['x'] - hero_x)**2 + (plot_attrs['y'] - hero_y)**2)
                    hero_related_attrs = plot_attrs.sort_values('DistToHero').head(5)['Label'].tolist()

            # --- PLOTTING ---
            fig = go.Figure()

            def get_style(label, is_brand=False):
                color = '#1f77b4' if is_brand else '#d62728'
                opacity = 1.0 if is_brand else 0.7
                if focus_brand != "None":
                    if is_brand:
                        if label == focus_brand: return color, 1.0
                        else: return '#d3d3d3', 0.2
                    else:
                        if label in hero_related_attrs: return color, 1.0
                        else: return '#d3d3d3', 0.2
                return color, opacity

            # Dots
            b_colors, b_opacities = [], []
            for _, row in plot_brands.iterrows():
                c, o = get_style(row['Label'], is_brand=True)
                b_colors.append(c); b_opacities.append(o)

            fig.add_trace(go.Scatter(
                x=plot_brands['x'], y=plot_brands['y'], mode='markers', name='Brands',
                marker=dict(size=10, color=b_colors, opacity=b_opacities, line=dict(width=1, color='white')),
                hoverinfo='text', hovertext=plot_brands['Label']
            ))

            a_colors, a_opacities = [], []
            for _, row in plot_attrs.iterrows():
                c, o = get_style(row['Label'], is_brand=False)
                a_colors.append(c); a_opacities.append(o)

            fig.add_trace(go.Scatter(
                x=plot_attrs['x'], y=plot_attrs['y'], mode='markers', name='Statements',
                marker=dict(size=7, color=a_colors, opacity=a_opacities),
                hoverinfo='text', hovertext=plot_attrs['Label']
            ))

            pass_colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
            for i, layer in enumerate(sel_passive_data):
                base_c = pass_colors[i % len(pass_colors)]
                l_colors = [base_c if focus_brand == "None" else '#d3d3d3' for _ in range(len(layer))]
                l_opacs = [0.9 if focus_brand == "None" else 0.2 for _ in range(len(layer))]
                fig.add_trace(go.Scatter(
                    x=layer['x'], y=layer['y'], mode='markers', name=layer['LayerName'].iloc[0],
                    marker=dict(size=9, symbol=layer['Shape'].iloc[0], color=l_colors, opacity=l_opacs, line=dict(width=1, color='white')),
                    hoverinfo='text', hovertext=layer['Label']
                ))

            # Labels
            annotations = []
            for i, row in plot_brands.iterrows():
                c, o = get_style(row['Label'], is_brand=True)
                if o > 0.3:
                    annotations.append(dict(
                        x=row['x'], y=row['y'], text=row['Label'],
                        showarrow=True, arrowhead=0, arrowcolor=c, ax=0, ay=-20,
                        font=dict(size=11, color=c, family="Arial Black"),
                        bgcolor="rgba(255,255,255,0.7)"
                    ))

            for i, row in plot_attrs.iterrows():
                c, o = get_style(row['Label'], is_brand=False)
                if o > 0.3:
                    annotations.append(dict(
                        x=row['x'], y=row['y'], text=row['Label'],
                        showarrow=True, arrowhead=0, arrowcolor=c, ax=0, ay=-15,
                        font=dict(size=11, color=c),
                        bgcolor="rgba(255,255,255,0.5)"
                    ))
            
            for i, layer in enumerate(sel_passive_data):
                base_c = pass_colors[i % len(pass_colors)]
                if focus_brand == "None":
                    for _, row in layer.iterrows():
                        annotations.append(dict(
                            x=row['x'], y=row['y'], text=row['Label'],
                            showarrow=True, arrowhead=0, arrowcolor=base_c, ax=0, ay=-15,
                            font=dict(size=11, color=base_c),
                            bgcolor="rgba(255,255,255,0.5)"
                        ))

            fig.update_layout(
                annotations=annotations,
                title={'text': f"Strategic Map: {focus_brand}" if focus_brand != "None" else "Strategic Perceptual Map", 'y':0.95, 'x':0.5, 'xanchor': 'center'},
                template="plotly_white", height=850,
                xaxis=dict(zeroline=True, showgrid=False, showticklabels=False),
                yaxis=dict(zeroline=True, showgrid=False, showticklabels=False),
                yaxis_scaleanchor="x", yaxis_scaleratio=1,
                dragmode='pan'
            )
            
            fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=10, fillcolor="blue", opacity=0.03, layer="below", line_width=0)
            fig.add_shape(type="rect", x0=-10, y0=-10, x1=0, y1=0, fillcolor="blue", opacity=0.03, layer="below", line_width=0)

            st.plotly_chart(fig, use_container_width=True, config={'editable': True, 'scrollZoom': True, 'displayModeBar': True})

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ==========================================
# TAB 2: INTERPRETATION
# ==========================================
with tab2:
    st.header("📝 Strategic Interpretation")
    
    if st.session_state.processed_data:
        df_attrs = st.session_state.df_attrs
        df_brands = st.session_state.df_brands
        passive_layers = st.session_state.passive_data
        
        # 1. ANALYZE AXES
        # We find the attribute with the max/min X and Y to define the axes
        x_max = df_attrs.loc[df_attrs['x'].idxmax()]['Label']
        x_min = df_attrs.loc[df_attrs['x'].idxmin()]['Label']
        y_max = df_attrs.loc[df_attrs['y'].idxmax()]['Label']
        y_min = df_attrs.loc[df_attrs['y'].idxmin()]['Label']
        
        st.subheader("1. The Core Conflicts")
        st.markdown(f"""
        The map is organized by two primary tensions in the market:
        * **Horizontal Axis:** Runs from **"{x_min}"** (Left) to **"{x_max}"** (Right).
        * **Vertical Axis:** Runs from **"{y_min}"** (Bottom) to **"{y_max}"** (Top).
        
        Brands located near these poles "own" that strategic territory.
        """)
        
        st.divider()
        
        # 2. ANALYZE PASSIVE LAYERS
        if passive_layers:
            st.subheader("2. Passive Layer Analysis")
            for layer in passive_layers:
                name = layer['LayerName'].iloc[0]
                st.markdown(f"#### Layer: {name}")
                
                # Find quadrant distribution
                q1 = len(layer[(layer['x'] > 0) & (layer['y'] > 0)])
                q2 = len(layer[(layer['x'] < 0) & (layer['y'] > 0)])
                q3 = len(layer[(layer['x'] < 0) & (layer['y'] < 0)])
                q4 = len(layer[(layer['x'] > 0) & (layer['y'] < 0)])
                
                # Determine dominant quadrant
                quads = {'Top-Right (Q1)': q1, 'Top-Left (Q2)': q2, 'Bottom-Left (Q3)': q3, 'Bottom-Right (Q4)': q4}
                dom_quad = max(quads, key=quads.get)
                
                st.write(f"- **Distribution:** Most items in this layer ({quads[dom_quad]} items) fall into the **{dom_quad}**.")
                
                # Find closest brand for the top item
                top_item = layer.sort_values('Distinctiveness', ascending=False).iloc[0]
                # Calculate distance to all brands
                df_brands['Dist'] = np.sqrt((df_brands['x'] - top_item['x'])**2 + (df_brands['y'] - top_item['y'])**2)
                closest_brand = df_brands.sort_values('Dist').iloc[0]['Label']
                
                st.write(f"- **Key Item:** *'{top_item['Label']}'* is the most distinct item in this layer. It is strategically closest to **{closest_brand}**.")
                st.markdown("---")
        else:
            st.info("No passive layers uploaded yet. Upload a file in the sidebar to see analysis here.")
            
    else:
        st.warning("👈 Please upload and process data in 'The Strategy Engine' tab first.")

# ==========================================
# TAB 3: DATA CLEANER
# ==========================================
with tab3:
    st.header("🧹 MRI Data Cleaner")
    st.markdown("Upload a raw MRI Crosstab (CSV, XLSX, XLS).")
    raw_mri = st.file_uploader("Upload Raw MRI", type=["csv", "xlsx", "xls"])
    
    if raw_mri:
        try:
            if raw_mri.name.endswith('.csv'): df_raw = pd.read_csv(raw_mri, header=None)
            else: df_raw = pd.read_excel(raw_mri, header=None)
                
            metric_row_idx = -1
            for i, row in df_raw.iterrows():
                if row.astype(str).str.contains("Weighted (000)", regex=False).any():
                    metric_row_idx = i
                    break
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
                df_clean = data_rows.iloc[:, cols]
                df_clean.columns = headers
                df_clean['Attitude'] = df_clean['Attitude'].astype(str).str.replace('General Attitudes: ', '', regex=False)
                df_clean = df_clean[df_clean['Attitude'].str.len() > 3]
                df_clean = df_clean[~df_clean['Attitude'].astype(str).str.contains("Study Universe|Total|Base|Sample", case=False, regex=True)]
                st.success("Cleaned!")
                st.download_button("Download CSV", df_clean.to_csv(index=False).encode('utf-8'), "Cleaned_MRI.csv", "text/csv")
            else: st.error("Could not find 'Weighted (000)' row.")
        except Exception as e: st.error(f"Error: {e}")
