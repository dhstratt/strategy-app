import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Universal Strategy Engine")

# --- MAIN APP ---
st.title("🧠 The Strategy Engine")
st.markdown("Upload your **Cleaned** CSV (Attributes in Col A, Brands in Cols B+).")
st.caption("💡 **Pro Tip:** You can CLICK and DRAG labels on the map to organize them! (Snapshot before refreshing)")

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
        
        if data_cols:
            
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
            # Brands (Bigger text, smaller dots)
            df_brands = pd.DataFrame(col_coords[:, :2], columns=['x', 'y'])
            df_brands['Label'] = cleaned_df.columns
            df_brands['Type'] = 'Brand'
            df_brands['Size'] = 12  # Requested: "Just a little smaller"
            
            # Attributes
            df_attrs = pd.DataFrame(row_coords[:, :2], columns=['x', 'y'])
            df_attrs['Label'] = cleaned_df.index
            df_attrs['Type'] = 'Attribute'
            df_attrs['Size'] = 6   # Attributes quite small
            
            # Strategic Importance
            df_attrs['Importance'] = np.sqrt(df_attrs['x']**2 + df_attrs['y']**2)
            
            # --- 6. UX CONTROLS ---
            st.sidebar.markdown("---")
            st.sidebar.subheader("🎨 Design Controls")
            
            total_attrs = len(df_attrs)
            n_show = st.sidebar.slider("Attribute Density", 
                                     min_value=5, max_value=total_attrs, value=15)
            
            show_attr_labels = st.sidebar.checkbox("Show Attribute Labels", value=True)
            
            # Filter Attributes
            top_attrs = df_attrs.sort_values('Importance', ascending=False).head(n_show)
            plot_data = pd.concat([df_brands, top_attrs])
            
            # Rounding
            plot_data['x']
