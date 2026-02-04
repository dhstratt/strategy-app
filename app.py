import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import collections

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Universal Strategy Engine")

# --- TABS ---
tab1, tab2 = st.tabs(["🧠 The Strategy Engine", "🧹 MRI Data Cleaner"])

# ==========================================
# TAB 1: THE STRATEGY ENGINE
# ==========================================
with tab1:
    st.title("🧠 The Strategy Engine")
    
    # --- LAYOUT: 2 Columns (Sidebar vs Main) ---
    col_nav, col_main = st.columns([1, 4])
    
    # --- SIDEBAR: DATA UPLOAD ---
    with st.sidebar:
        st.header("📂 Data Manager")
        uploaded_file = st.file_uploader("1. Upload Core Data", type=["csv", "xlsx"], key="active")
        passive_files = st.file_uploader("2. Upload Passive Layers", type=["csv", "xlsx"], accept_multiple_files=True, key="passive")
        st.divider()

    # --- HELPER FUNCTIONS ---
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
        df = df[~df.index.astype(str).str.contains("Total|Universe|Base|Sample", case=False, regex=True)]
        valid_cols = [c for c in df.columns if "total" not in str(c).lower() and "base" not in str(c).lower()]
        return df[valid_cols]

    if uploaded_file is not None:
        try:
            # --- 1. PROCESS CORE DATA ---
            df_active_raw = load_file(uploaded_file)
            
            # Initial Clean & Math
            df_math = clean_df(df_active_raw)
            df_math = df_math.loc[(df_math != 0).any(axis=1)] 
            df_math = df_math.loc[:, (df_math != 0).any(axis=0)]
            
            # SVD Calculation
            N = df_math.values
            P = N / N.sum()
            r = P.sum(axis=1)
            c = P.sum(axis=0)
            E = np.outer(r, c)
            E[E < 1e-9] = 1e-9
            R = (P - E) / np.sqrt(E)
            U, s, Vh = np.linalg.svd(R, full_matrices=False)
            
            # --- CALCULATE MAP ACCURACY ---
            inertia = s**2
            total_inertia = np.sum(inertia)
            explained_inertia = np.sum(inertia[:2])
            map_accuracy = (explained_inertia / total_inertia) * 100
            
            # Coordinates
            row_coords = (U * s) / np.sqrt(r[:, np.newaxis])
            col_coords = (Vh.T * s) / np.sqrt(c[:, np.newaxis])
            
            # Master DataFrames
            df_brands = pd.DataFrame(col_coords[:, :2], columns=['x', 'y'])
            df_brands['Label'] = df_math.columns
            df_brands['Type'] = 'Brand (Core)'
            df_brands['Distinctiveness'] = np.sqrt(df_brands['x']**2 + df_brands['y']**2)
            
            df_attrs = pd.DataFrame(row_coords[:, :2], columns=['x', 'y'])
            df_attrs['Label'] = df_math.index
            df_attrs['Type'] = 'Attribute'
            df_attrs['Distinctiveness'] = np.sqrt(df_attrs['x']**2 + df_attrs['y']**2)

            # --- 2. PROCESS PASSIVE LAYERS ---
            passive_layer_data = []
            if passive_files:
                for p_file in passive_files:
                    try:
                        p_raw = load_file(p_file)
                        p_clean = clean_df(p_raw)
                        common_brands = list(set(p_clean.columns) & set(df_math.columns))
                        common_attrs = list(set(p_clean.index) & set(df_math.index))
                        
                        if len(common_brands) > len(common_attrs): # Passive Attributes
                            p_aligned = p_clean[common_brands].reindex(columns=df_math.columns).fillna(0)
                            p_profiles = p_aligned.div(p_aligned.sum(axis=1).replace(0,1), axis=0)
                            proj = p_profiles.values @ col_coords[:, :2] / s[:2]
                            res = pd.DataFrame(proj, columns=['x', 'y'])
                            res['Label'] = p_aligned.index
                            res['LayerName'] = p_file.name
                            res['Shape'] = 'star'
                        else: # Passive Brands
                            p_aligned = p_clean.reindex(df_math.index).fillna(0)
                            p_profiles = p_aligned.div(p_aligned.sum(axis=0).replace(0,1), axis=1)
                            proj = p_profiles.T.values @ row_coords[:, :2] / s[:2]
                            res = pd.DataFrame(proj, columns=['x', 'y'])
                            res['Label'] = p_aligned.columns
                            res['LayerName'] = p_file.name
                            res['Shape'] = 'diamond'

                        res['Distinctiveness'] = np.sqrt(res['x']**2 + res['y']**2)
                        passive_layer_data.append(res)
                    except: pass

            # --- 3. SIDEBAR CONTROL CENTER ---
            with st.sidebar:
                st.header("🎯 Map Controls")
                
                # METRIC: Map Accuracy
                st.metric("Map Stability", f"{map_accuracy:.1f}%", help="Percentage of variance explained by these 2 dimensions. Higher is better.")
                
                # A. Core Brands Control
                with st.expander("🔹 Core Brands", expanded=True):
                    show_all_brands = st.checkbox("Show All Brands", value=True)
                    if not show_all_brands:
                        all_b = sorted(df_brands['Label'].tolist())
                        sel_brands = st.multiselect("Filter Brands (Searchable):", all_b, default=all_b)
                    else:
                        sel_brands = df_brands['Label'].tolist()

                # B. Core Statements Control
                with st.expander("🔹 Core Statements", expanded=True):
                    show_all_attrs = st.checkbox("Show All Statements", value=False)
                    if not show_all_attrs:
                        # Default to top 15 distinctive for cleanliness
                        all_a = sorted(df_attrs['Label'].tolist())
                        top_15 = df_attrs.sort_values('Distinctiveness', ascending=False).head(15)['Label'].tolist()
                        sel_attrs = st.multiselect("Filter Statements (Searchable):", all_a, default=top_15)
                    else:
                        sel_attrs = df_attrs['Label'].tolist()

                # C. Passive Layers Control
                sel_passive_data = []
                if passive_layer_data:
                    st.subheader("🔸 Passive Layers")
                    for layer in passive_layer_data:
                        lname = layer['LayerName'].iloc[0]
                        with st.expander(f"{lname}", expanded=False):
                            show_all_pass = st.checkbox(f"Show All: {lname}", value=True)
                            if not show_all_pass:
                                all_l = sorted(layer['Label'].tolist())
                                sel_l = st.multiselect("Filter Items:", all_l, default=all_l)
                            else:
                                sel_l = layer['Label'].tolist()
                            
                            if sel_l:
                                sel_passive_data.append(layer[layer['Label'].isin(sel_l)])

            # --- 4. FILTER DATA ---
            plot_brands = df_brands[df_brands['Label'].isin(sel_brands)]
            plot_attrs = df_attrs[df_attrs['Label'].isin(sel_attrs)]

            # --- 5. PLOT ---
            fig = go.Figure()

            # Core Brands
            fig.add_trace(go.Scatter(
                x=plot_brands['x'], y=plot_brands['y'], mode='markers+text',
                text=plot_brands['Label'], textposition="top center", name="Brands",
                marker=dict(size=12, color='#1f77b4', line=dict(width=1, color='white')),
                textfont=dict(family="Arial Black", size=11, color='#1f77b4')
            ))

            # Core Attributes
            fig.add_trace(go.Scatter(
                x=plot_attrs['x'], y=plot_attrs['y'], mode='markers+text',
                text=plot_attrs['Label'], textposition="top center", name="Statements",
                marker=dict(size=8, color='#d62728', opacity=0.7),
                textfont=dict(size=10, color='#d62728')
            ))

            # Passive Layers
            colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
            for i, layer in enumerate(sel_passive_data):
                c = colors[i % len(colors)]
                sym = layer['Shape'].iloc[0]
                fig.add_trace(go.Scatter(
                    x=layer['x'], y=layer['y'], mode='markers+text',
                    text=layer['Label'], textposition="top center", name=layer['LayerName'].iloc[0],
                    marker=dict(size=10, symbol=sym, color=c, opacity=0.9, line=dict(width=1, color='white')),
                    textfont=dict(size=10, color=c)
                ))

            # --- 6. UX & ZOOM ---
            fig.update_layout(
                title={'text': "Strategic Perceptual Map", 'y':0.95, 'x':0.5, 'xanchor': 'center'},
                template="plotly_white", height=850,
                xaxis=dict(zeroline=True, showgrid=False, showticklabels=False, zerolinecolor='#E5E5E5'),
                yaxis=dict(zeroline=True, showgrid=False, showticklabels=False, zerolinecolor='#E5E5E5'),
                # LOCK ASPECT RATIO 1:1
                yaxis_scaleanchor="x", yaxis_scaleratio=1,
                dragmode='pan', hovermode='closest'
            )
            
            # Quadrants
            fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=10, fillcolor="blue", opacity=0.03, layer="below", line_width=0)
            fig.add_shape(type="rect", x0=-10, y0=-10, x1=0, y1=0, fillcolor="blue", opacity=0.03, layer="below", line_width=0)

            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ==========================================
# TAB 2: DATA CLEANER
# ==========================================
with tab2:
    st.header("🧹 MRI Data Cleaner")
    st.markdown("Upload a raw MRI Crosstab (CSV) to clean it for the map.")
    raw_mri = st.file_uploader("Upload Raw MRI", type=["csv"])
    
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
                        if "Study Universe" not in brand and "Total" not in brand and brand != 'nan':
                            cols.append(c)
                            headers.append(brand)
                
                df_clean = data_rows.iloc[:, cols]
                df_clean.columns = headers
                df_clean['Attitude'] = df_clean['Attitude'].astype(str).str.replace('General Attitudes: ', '', regex=False)
                df_clean = df_clean[df_clean['Attitude'].str.len() > 3]
                
                st.success("File Cleaned!")
                st.download_button("Download Clean CSV", df_clean.to_csv(index=False).encode('utf-8'), "Cleaned_MRI.csv", "text/csv")
            else:
                st.error("Could not find 'Weighted (000)' row.")
        except Exception as e:
            st.error(f"Error: {e}")
