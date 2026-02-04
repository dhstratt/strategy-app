import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2_contingency

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="Universal Strategy Engine")
st.title("🧠 The Strategy Engine")
st.markdown("Upload your crosstab (Brands x Attributes) to generate a Perceptual Map instantly.")

# --- STEP 1: UPLOAD DATA ---
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.sidebar.success("Data Uploaded!")
    
    # --- STEP 2: MAP COLUMNS ---
    all_columns = df.columns.tolist()
    st.sidebar.subheader("🗺️ Map Your Data")
    brand_col = st.sidebar.selectbox("Which column has Brand Names?", all_columns, index=0)
    
    # Default to all numeric columns
    default_cols = [c for c in all_columns if c != brand_col and df[c].dtype in ['float64', 'int64']]
    attr_cols = st.sidebar.multiselect("Select Attribute Columns", default_cols, default=default_cols)
    
    if st.sidebar.button("Run Analysis"):
        try:
            # Prepare Data Matrix
            data_matrix = df.set_index(brand_col)[attr_cols]
            data_matrix = data_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # --- STEP 3: THE MATH ENGINE (CA) ---
            N = data_matrix.values
            grand_total = N.sum()
            P = N / grand_total
            
            row_sums = P.sum(axis=1)
            col_sums = P.sum(axis=0)
            
            # Expected & Residuals
            E = np.outer(row_sums, col_sums)
            R = (P - E) / np.sqrt(E)
            
            # SVD (Singular Value Decomposition)
            U, s, Vh = np.linalg.svd(R, full_matrices=False)
            
            # --- STEP 4: STABILITY CHECK ---
            eigenvalues = s**2
            total_inertia = np.sum(eigenvalues)
            explained_variance = eigenvalues / total_inertia
            total_var_2d = (explained_variance[0] + explained_variance[1]) * 100
            
            st.divider()
            col_metrics, col_warn = st.columns([1, 2])
            with col_metrics:
                st.metric("Map Accuracy", f"{total_var_2d:.1f}%")
            
            with col_warn:
                if total_var_2d < 60:
                    st.error("⚠️ CRITICAL WARNING: MAP UNSTABLE. Relationships may be misleading.")
                else:
                    st.success("✅ MAP STABLE. Reliable for strategic planning.")

            # --- STEP 5: CALCULATE COORDINATES ---
            # Brands (Rows)
            row_coords = (U * s) / np.sqrt(row_sums[:, np.newaxis])
            brand_df = pd.DataFrame(row_coords[:, :2], columns=['Dim1', 'Dim2'])
            brand_df['Label'] = data_matrix.index
            brand_df['Type'] = 'Brand'
            
            # Attributes (Columns)
            col_coords = (Vh.T * s) / np.sqrt(col_sums[:, np.newaxis])
            attr_df = pd.DataFrame(col_coords[:, :2], columns=['Dim1', 'Dim2'])
            attr_df['Label'] = data_matrix.columns
            attr_df['Type'] = 'Attribute'
            
            # Combine for plotting
            plot_df = pd.concat([brand_df, attr_df])
            
            # --- STEP 6: WHITE SPACE FINDER ---
            # Calculate distance from every attribute to nearest brand
            isolation_scores = []
            for _, attr_row in attr_df.iterrows():
                attr_pos = np.array([attr_row['Dim1'], attr_row['Dim2']])
                dists = [np.linalg.norm(attr_pos - np.array([row['Dim1'], row['Dim2']])) for _, row in brand_df.iterrows()]
                isolation_scores.append(min(dists))
            
            attr_df['Isolation_Score'] = isolation_scores
            top_opps = attr_df.sort_values('Isolation_Score', ascending=False).head(3)
            
            # --- STEP 7: VISUALIZATION ---
            col_map, col_insight = st.columns([3, 1])
            
            with col_map:
                fig = px.scatter(plot_df, x='Dim1', y='Dim2', text='Label', color='Type',
                                title="Strategic Landscape", template="plotly_white",
                                color_discrete_map={'Brand': '#1f77b4', 'Attribute': '#d62728'},
                                height=700)
                fig.update_traces(textposition='top center', marker_size=12)
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                fig.add_vline(x=0, line_dash="dot", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
            
            with col_insight:
                st.subheader("🚀 Innovation Gaps")
                st.write("Top 'White Space' Opportunities:")
                for i, row in top_opps.iterrows():
                    st.info(f"**{row['Label']}**\n(Gap Score: {row['Isolation_Score']:.2f})")

        except Exception as e:
            st.error(f"Error: {e}. Please ensure your data is clean (numeric values only).")
else:
    st.info("👈 Waiting for data... Upload your file in the sidebar.")
