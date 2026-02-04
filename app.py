import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import collections
import io

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Universal Strategy Engine")

# --- TABS ---
tab1, tab2 = st.tabs(["🧠 The Strategy Engine", "🧹 MRI Data Cleaner"])

# ==========================================
# TAB 1: THE STRATEGY ENGINE
# ==========================================
with tab1:
    st.title("🧠 The Strategy Engine")
    
    col_nav, col_main = st.columns([1, 4])
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("📂 Data Manager")
        uploaded_file = st.file_uploader("1. Upload Core Data", type=["csv", "xlsx"], key="active")
        passive_files = st.file_uploader("2. Upload Passive Layers", type=["csv", "xlsx"], accept_multiple_files=True, key="passive")
        st.divider()

    def load_file(file):
        if file.name.endswith('.csv'): return pd.read_csv(file)
        else: return pd.read_excel(file)

    def clean_df(df):
        # 1. Set Index
        label_col = df.columns[0]
        df = df.set_index(label_col)
        
        # 2. Clean Numbers (remove commas)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce')
        df = df.fillna(0)
        
        # 3. THE ASSASSIN: Remove "Study Universe" specifically from Rows
        # We also remove Total, Base, Sample to be safe
        df = df[~df.index.astype(str).str.contains("Study Universe|Total|Base|Sample", case=False, regex=True)]
        
        # 4. THE ASSASSIN: Remove "Study Universe" specifically from Columns
        valid_cols = [c for c in df.columns if 
                      "study universe" not in str(c).lower() and 
                      "total" not in str(c).lower() and 
                      "base" not in str(c).lower()]
        
        return df[valid_cols]

    if uploaded_file is not None:
        try:
            # --- PROCESS DATA ---
            df_active_raw = load_file(uploaded_file)
            df_math = clean_df(df_active_raw)
            
            # Final Safety Check: Drop empty rows/cols
            df_math = df_math.loc[(df_math != 0).any(axis=1)] 
            df_math = df_math.loc[:, (df_math != 0).any(axis=0)]
            
            if df_math.empty:
                st.error("Error: All data was filtered out. Check if your file only contains 'Study Universe' or empty data.")
                st.stop()

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

            # --- CONTROLS ---
            with st.sidebar:
                st.header("🎯 Map Controls")
                st.metric("Map Stability", f"{map_accuracy:.1f}%")
                
                with st.expander("🔹 Core Brands", expanded=True):
                    if not st.checkbox("Show All Brands", value=True):
                        all_b = sorted(df_brands['Label'].tolist())
                        sel_brands = st.multiselect("Filter:", all_b, default=all_b)
                    else: sel_brands = df_brands['Label'].tolist()

                with st.expander("🔹 Core Statements", expanded=True):
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
                        with st.expander(f"{lname}", expanded=False):
                            if not st.checkbox(f"Show All: {lname}", value=True):
                                all_l = sorted(layer['Label'].tolist())
                                sel_l = st.multiselect("Filter:", all_l, default=all_l)
                            else: sel_l = layer['Label'].tolist()
                            if sel_l: sel_passive_data.append(layer[layer['Label'].isin(sel_l)])

            # --- PLOTTING ---
            plot_brands = df_brands[df_brands['Label'].isin(sel_brands)]
            plot_attrs = df_attrs[df_attrs['Label'].isin(sel_attrs)]

            fig = go.Figure()

            # 1. ADD DOTS
            # Core Brands
            fig.add_trace(go.Scatter(
                x=plot_brands['x'], y=plot_brands['y'], mode='markers', name='Brands',
                marker=dict(size=10, color='#1f77b4', line=dict(width=1, color='white')), 
                hoverinfo='text', hovertext=plot_brands['Label']
            ))

            # Core Attributes
            fig.add_trace(go.Scatter(
                x=plot_attrs['x'], y=plot_attrs['y'], mode='markers', name='Statements',
                marker=dict(size=7, color='#d62728', opacity=0.7),
                hoverinfo='text', hovertext=plot_attrs['Label']
            ))

            # Passive Layers
            pass_colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
            for i, layer in enumerate(sel_passive_data):
                c = pass_colors[i % len(pass_colors)]
                fig.add_trace(go.Scatter(
                    x=layer['x'], y=layer['y'], mode='markers', name=layer['LayerName'].iloc[0],
                    marker=dict(size=9, symbol=layer['Shape'].iloc[0], color=c, opacity=0.9, line=dict(width=1, color='white')),
                    hoverinfo='text', hovertext=layer['Label']
                ))

            # 2. ADD DRAGGABLE ANNOTATIONS
            annotations = []
            
            # Brand Labels
            for _, row in plot_brands.iterrows():
                annotations.append(dict(
                    x=row['x'], y=row['y'], text=row['Label'],
                    showarrow=True, arrowhead=0, arrowcolor='#1f77b4', ax=0, ay=-20,
                    font=dict(size=11, color='#1f77b4', family="Arial Black"),
                    bgcolor="rgba(255,255,255,0.7)"
                ))

            # Attribute Labels
            for _, row in plot_attrs.iterrows():
                annotations.append(dict(
                    x=row['x'], y=row['y'], text=row['Label'],
                    showarrow=True, arrowhead=0, arrowcolor='#d62728', ax=0, ay=-15,
                    font=dict(size=11, color='#d62728'), 
                    bgcolor="rgba(255,255,255,0.5)"
                ))

            # Passive Labels
            for i, layer in enumerate(sel_passive_data):
                c = pass_colors[i % len(pass_colors)]
                for _, row in layer.iterrows():
                    annotations.append(dict(
                        x=row['x'], y=row['y'], text=row['Label'],
                        showarrow=True, arrowhead=0, arrowcolor=c, ax=0, ay=-15,
                        font=dict(size=11, color=c),
                        bgcolor="rgba(255,255,255,0.5)"
                    ))

            fig.update_layout(
                annotations=annotations,
                title={'text': "Strategic Perceptual Map", 'y':0.95, 'x':0.5, 'xanchor': 'center'},
                template="plotly_white", height=850,
                xaxis=dict(zeroline=True, showgrid=False, showticklabels=False),
                yaxis=dict(zeroline=True, showgrid=False, showticklabels=False),
                yaxis_scaleanchor="x", yaxis_scaleratio=1,
                dragmode='pan'
            )
            
            # Add Quadrant Lines
            fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=10, fillcolor="blue", opacity=0.03, layer="below", line_width=0)
            fig.add_shape(type="rect", x0=-10, y0=-10, x1=0, y1=0, fillcolor="blue", opacity=0.03, layer="below", line_width=0)

            st.plotly_chart(fig, use_container_width=True, config={'editable': True, 'scrollZoom': True, 'displayModeBar': True})

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ==========================================
# TAB 2: DATA CLEANER
# ==========================================
with tab2:
    st.header("🧹 MRI Data Cleaner")
    raw_mri = st.file_uploader("Upload Raw MRI CSV", type=["csv"])
    if raw_mri:
        try:
            df_raw = pd.read_csv(raw_mri, header=None)
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
                        # EXPLICITLY BLOCK STUDY UNIVERSE HERE TOO
                        if "Study Universe" not in brand and "Total" not in brand and brand != 'nan':
                            cols.append(c); headers.append(brand)
                df_clean = data_rows.iloc[:, cols]
                df_clean.columns = headers
                df_clean['Attitude'] = df_clean['Attitude'].astype(str).str.replace('General Attitudes: ', '', regex=False)
                df_clean = df_clean[df_clean['Attitude'].str.len() > 3]
                st.success("Cleaned!")
                st.download_button("Download CSV", df_clean.to_csv(index=False).encode('utf-8'), "Cleaned_MRI.csv", "text/csv")
            else: st.error("Could not find 'Weighted (000)' row.")
        except Exception as e: st.error(f"Error: {e}")
