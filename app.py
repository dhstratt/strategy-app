import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Universal Strategy Engine")

# --- MAIN APP ---
st.title("🧠 The Strategy Engine")
st.markdown("Upload your **Cleaned** CSV (Attributes in Col A, Brands in Cols B+).")

# 1. UPLOAD
uploaded_file = st.sidebar.file_uploader("Upload Clean CSV/Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Load Data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.sidebar.success("Data Loaded!")
        
        # 2. SELECT COLUMNS
        label_col = df.columns[0] 
        all_brands = df.columns[1:].tolist()
        data_cols = st.sidebar.multiselect("Select Brands to Map", all_brands, default=all_brands)
        
        if st.sidebar.button("Run Analysis") and data_cols:
            
            # 3. PREPARE DATA
            cleaned_df = df.set_index(label_col)[data_cols]
            
            # Clean commas/numbers
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce')
            
            cleaned_df = cleaned_df.fillna(0)
            cleaned_df = cleaned_df.loc[(cleaned_df != 0).any(axis=1)]
            cleaned_df = cleaned_df.loc[:, (cleaned_df != 0).any(axis=0)]
            
            if cleaned_df.empty:
                st.error("Error: Data is empty.")
                st.stop()

            # 4. THE MATH (SVD)
            N = cleaned_df.values
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
            
            # 5. PREPARE VISUALIZATION DATAFRAMES
            # Brands
            df_brands = pd.DataFrame(col_coords[:, :2], columns=['x', 'y'])
            df_brands['Label'] = cleaned_df.columns
            df_brands['Type'] = 'Brand'
            df_brands['Size'] = 15
            
            # Attributes
            df_attrs = pd.DataFrame(row_coords[:, :2], columns=['x', 'y'])
            df_attrs['Label'] = cleaned_df.index
            df_attrs['Type'] = 'Attribute'
            df_attrs['Size'] = 8
            
            # Strategic Importance (Distance from Center)
            df_attrs['Importance'] = np.sqrt(df_attrs['x']**2 + df_attrs['y']**2)
            
            # --- 6. UX CONTROLS ---
            st.sidebar.markdown("---")
            st.sidebar.subheader("🎨 Design Controls")
            
            total_attrs = len(df_attrs)
            n_show = st.sidebar.slider("Attribute Density (Most Important First)", 
                                     min_value=5, max_value=total_attrs, value=15)
            
            show_attr_labels = st.sidebar.checkbox("Show Attribute Labels", value=True)
            
            # --- FILTER LOGIC ---
            top_attrs = df_attrs.sort_values('Importance', ascending=False).head(n_show)
            plot_data = pd.concat([df_brands, top_attrs])
            
            # Rounding
            plot_data['x'] = plot_data['x'].round(2)
            plot_data['y'] = plot_data['y'].round(2)
            
            # --- SMART LABEL LOGIC (THE FIX) ---
            # Always create the SmartLabel column so Plotly doesn't crash
            if show_attr_labels:
                # If checked, SmartLabel is just the Label
                plot_data['SmartLabel'] = plot_data['Label']
            else:
                # If unchecked, SmartLabel is blank for attributes
                plot_data['SmartLabel'] = plot_data.apply(
                    lambda row: row['Label'] if row['Type'] == 'Brand' else '', axis=1
                )

            # Accuracy
            inertia = s**2
            accuracy = (np.sum(inertia[:2]) / np.sum(inertia)) * 100
            
            st.divider()
            col1, col2 = st.columns([1, 3])
            col1.metric("Map Accuracy", f"{accuracy:.1f}%")
            col2.caption(f"Showing the top {n_show} most differentiating attributes out of {total_attrs}.")
            
            # The Chart
            fig = px.scatter(plot_data, x='x', y='y', text='SmartLabel', color='Type',
                            size='Size', size_max=15,
                            title="Strategic Perceptual Map (Clean View)", 
                            template="plotly_white", height=800,
                            color_discrete_map={'Brand': '#1f77b4', 'Attribute': '#d62728'},
                            # Now safe because SmartLabel always exists
                            hover_data={'Label': True, 'x':':.2f', 'y':':.2f', 'Size':False, 'SmartLabel':False})
            
            fig.update_traces(textposition='top center')
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray', showgrid=False),
                yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray', showgrid=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 7. OPPORTUNITY FINDER
            st.subheader("🔭 Opportunity Finder (Top 5)")
            
            scores = []
            for _, attr in df_attrs.iterrows():
                dists = np.linalg.norm(df_brands[['x','y']].values - attr[['x','y']].values, axis=1)
                scores.append(dists.min())
            
            df_attrs['Isolation'] = scores
            df_attrs['Isolation'] = df_attrs['Isolation'].round(2)
            
            st.table(df_attrs.sort_values('Isolation', ascending=False).head(5)[['Label', 'Isolation']])
            
    except Exception as e:
        st.error(f"Something went wrong: {e}")
