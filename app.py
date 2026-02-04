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
    st.markdown("Upload **Cleaned** Data below.")

    # --- SIDEBAR: DATA UPLOAD ---
    st.sidebar.header("📂 Data Manager")
    uploaded_file = st.sidebar.file_uploader("1. Upload Core Data", type=["csv", "xlsx"], key="active")
    passive_files = st.sidebar.file_uploader("2. Upload Passive Layers", type=["csv", "xlsx"], accept_multiple_files=True, key="passive")

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
        # --- 1. PROCESS CORE DATA ---
        try:
            df_active_raw = load_file(uploaded_file)
            st.sidebar.success(f"Loaded: {uploaded_file.name}")
            
            # Initial Clean
            label_col = df_active_raw.columns[0]
            # We need all brands initially to run the math correctly
            all_brands = df_active_raw.columns[1:].tolist()
            
            # Run Math on EVERYTHING first (Mathematical Integrity)
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
            
            # Coordinates
            row_coords = (U * s) / np.sqrt(r[:, np.newaxis])
            col_coords = (Vh.T * s) / np.sqrt(c[:, np.newaxis])
            
            # Create Master DataFrames
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
                        
                        # Logic A: Passive Attributes (Rows)
                        if len(common_brands) > len(common_attrs):
                            p_aligned = p_clean[common_brands].reindex(columns=df_math.columns).fillna(0)
                            row_sums = p_aligned.sum(axis=1)
                            row_sums[row_sums == 0] = 1
                            p_profiles = p_aligned.div(row_sums, axis=0)
                            proj = p_profiles.values @ col_coords[:, :2]
                            proj = proj / s[:2]
                            
                            res = pd.DataFrame(proj, columns=['x', 'y'])
                            res['Label'] = p_aligned.index
                            res['Type'] = f"Layer: {p_file.name}"
                            res['LayerName'] = p_file.name
                            res['Shape'] = 'star'
                            
                        # Logic B: Passive Brands (Columns)
                        else:
                            p_aligned = p_clean.reindex(df_math.index).fillna(0)
                            col_sums = p_aligned.sum(axis=0)
                            col_sums[col_sums == 0] = 1
                            p_profiles = p_aligned.div(col_sums, axis=1)
                            proj = p_profiles.T.values @ row_coords[:, :2]
                            proj = proj / s[:2]
                            
                            res = pd.DataFrame(proj, columns=['x', 'y'])
                            res['Label'] = p_aligned.columns
                            res['Type'] = f"Layer: {p_file.name}"
                            res['LayerName'] = p_file.name
                            res['Shape'] = 'diamond'

                        res['Distinctiveness'] = np.sqrt(res['x']**2 + res['y']**2)
                        passive_layer_data.append(res)
                    except Exception as e:
                        st.sidebar.error(f"Error in {p_file.name}: {e}")

            # --- 3. THE CONTROL CENTER (Sidebar UX) ---
            st.sidebar.markdown("---")
            st.sidebar.header("🎨 Map Layers")
            
            # A. Core Layer Control
            with st.sidebar.expander("🔹 Core Map (Active)", expanded=True):
                # 1. Brands
                all_brand_labels = sorted(df_brands['Label'].tolist())
                # Default: All brands usually
                sel_brands = st.multiselect("Select Brands:", all_brand_labels, default=all_brand_labels)
                
                # 2. Statements
                all_attr_labels = sorted(df_attrs['Label'].tolist())
                # Default: Top 15 distinct
                top_15_attrs = df_attrs.sort_values('Distinctiveness', ascending=False).head(15)['Label'].tolist()
                sel_attrs = st.multiselect("Select Statements:", all_attr_labels, default=top_15_attrs)
            
            # B. Passive Layer Controls
            sel_passive_data = []
            if passive_layer_data:
                st.sidebar.markdown("---")
                st.sidebar.subheader("🔸 Passive Layers")
                
                for layer in passive_layer_data:
                    layer_name = layer['LayerName'].iloc[0]
                    with st.sidebar.expander(f"{layer_name}", expanded=False):
                        # Sort alphabetical
                        l_labels = sorted(layer['Label'].tolist())
                        # Default: Top 10 to keep it clean.
                        l_top = layer.sort_values('Distinctiveness', ascending=False).head(10)['Label'].tolist()
                        
                        sel_l_items = st.multiselect(f"Show items from {layer_name}:", l_labels, default=l_top)
                        
                        # Filter this layer's dataframe
                        if sel_l_items:
                            filtered_layer = layer[layer['Label'].isin(sel_l_items)]
                            sel_passive_data.append(filtered_layer)

            # --- 4. FILTER DATA FOR PLOTTING ---
            plot_brands = df_brands[df_brands['Label'].isin(sel_brands)]
            plot_attrs = df_attrs[df_attrs['Label'].isin(sel_attrs)]

            # --- 5. PLOT ---
            fig = go.Figure()

            # Core Brands (Blue Circles)
            fig.add_trace(go.Scatter(
                x=plot_brands['x'], y=plot_brands['y'],
                mode='markers+text', name='Brands',
                text=plot_brands['Label'], textposition="top center",
                marker=dict(size=12, color='#1f77b4', line=dict(width=1, color='white')),
                textfont=dict(size=11, color='#1f77b4', family="Arial Black")
            ))

            # Core Attributes (Red Circles)
            fig.add_trace(go.Scatter(
                x=plot_attrs['x'], y=plot_attrs['y'],
                mode='markers+text', name='Statements',
                text=plot_attrs['Label'], textposition="top center",
                marker=dict(size=8, color='#d62728', opacity=0.7),
                textfont=dict(size=10, color='#d62728')
            ))

            # Passive Layers (Colored Stars/Diamonds)
            passive_colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
            for i, layer in enumerate(sel_passive_data):
                l_color = passive_colors[i % len(passive_colors)]
                symbol = layer['Shape'].iloc[0] # diamond or star
                
                fig.add_trace(go.Scatter(
                    x=layer['x'], y=layer['y'],
                    mode='markers+text', name=layer['LayerName'].iloc[0],
                    text=layer['Label'], textposition="top center",
                    marker=dict(size=10, symbol=symbol, color=l_color, opacity=0.9, line=dict(width=1, color='white')),
                    textfont=dict(size=10, color=l_color)
                ))

            # --- 6. UX & ZOOM IMPROVEMENTS ---
            fig.update_layout(
                title={'text': "Strategic Perceptual Map", 'y':0.95, 'x':0.5, 'xanchor': 'center'},
                template="plotly_white",
                height=800,
                # Clean Axes (No Interpretation)
                xaxis=dict(zeroline=True, showgrid=False, showticklabels=False),
                yaxis=dict(zeroline=True, showgrid=False, showticklabels=False),
                # ZOOM PHYSICS: Locking Aspect Ratio
                yaxis_scaleanchor="x", 
                yaxis_scaleratio=1,
                dragmode='pan', 
                hovermode='closest'
            )

            # Add Quadrants
            fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=10, fillcolor="blue", opacity=0.03, layer="below", line_width=0)
            fig.add_shape(type="rect", x0=-10, y0=-10, x1=0, y1=0, fillcolor="blue", opacity=0.03, layer="below", line_width=0)

            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})

        except Exception as e:
            st.error(f"Error processing data: {e}")

# ==========================================
# TAB 2: DATA CLEANER
# ==========================================
with tab2:
    st.header("🧹 MRI Data Cleaner")
    st.markdown("Upload a messy MRI Crosstab (CSV), and this will clean it for the map.")
    
    raw_mri = st.file_uploader("Upload Raw MRI CSV", type=["csv"])
    
    if raw_mri:
        try:
            # Simple cleaning logic from before
            df_raw = pd.read_csv(raw_mri, header=None)
            
            # Find Weighted row
            metric_row_idx = -1
            for i, row in df_raw.iterrows():
                if row.astype(str).str.contains("Weighted (000)", regex=False).any():
                    metric_row_idx = i
                    break
            
            if metric_row_idx != -1:
                brand_row = df_raw.iloc[metric_row_idx - 1]
                metric_row = df_raw.iloc[metric_row_idx]
                data_rows = df_raw.iloc[metric_row_idx + 1:].copy()
                
                cols = [0]
                headers = ['Attitude']
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
                st.dataframe(df_clean.head())
                
                # Download Button
                csv = df_clean.to_csv(index=False).encode('utf-8')
                st.download_button("Download Clean CSV", csv, "Cleaned_MRI_Data.csv", "text/csv")
            else:
                st.error("Could not find 'Weighted (000)' row.")
                
        except Exception as e:
            st.error(f"Error cleaning file: {e}")
