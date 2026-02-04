import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2_contingency

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Universal Strategy Engine")

# --- HELPER: ADJUSTABLE LOADER ---
def load_data(uploaded_file, header_row_idx):
    """
    Loads data with a user-specified header row.
    Handles encoding issues common with MRI exports.
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            # Try default encoding, then fallback to 'latin1' (common for Excel CSVs)
            try:
                df = pd.read_csv(uploaded_file, header=header_row_idx, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, header=header_row_idx, encoding='latin1')
        else:
            df = pd.read_excel(uploaded_file, header=header_row_idx)
            
        # Clean up empty rows/cols
        df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)
        
        # Filter out MRI footer garbage
        # We assume the first column is the label column
        if not df.empty:
            first_col = df.columns[0]
            # Convert to string to avoid errors, filter out copyright/metadata text
            df = df[~df[first_col].astype(str).str.contains("Exported from|MRI-Simmons|Copyright", case=False, na=False)]
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

# --- MAIN APP ---
st.title("🧠 The Strategy Engine")

# STEP 1: UPLOAD
uploaded_file = st.sidebar.file_uploader("Upload Crosstab", type=["csv", "xlsx"])

if uploaded_file is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Data Calibration")
    
    # MANUAL OVERRIDE: Let the user adjust the header row
    header_idx = st.sidebar.number_input(
        "Header Row Number (Adjust if columns look wrong)", 
        min_value=0, max_value=30, value=11, step=1
    )
    
    # Load Data
    df = load_data(uploaded_file, header_idx)
    
    if not df.empty:
        # DATA DEBUGGER: Show the user what we loaded
        with st.expander("👀 View Raw Data (Click to Expand)", expanded=True):
            st.write(f"Showing first 5 rows (using Header Row {header_idx}):")
            st.dataframe(df.head(5))
            st.caption("If the column headers look like 'Unnamed' or 'Study Name', increase the Header Row Number in the sidebar.")

        # STEP 2: COLUMN MAPPING
        all_cols = df.columns.tolist()
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🗺️ Column Mapping")
        
        # 1. Row Labels
        label_col = st.sidebar.selectbox("Column with Statements/Attributes:", all_cols, index=0)
        
        # 2. Data Columns (Auto-detect "(000)")
        potential_data_cols = [c for c in all_cols if c != label_col]
        # Heuristic: looking for (000) or 'Weighted'
        default_data = [c for c in potential_data_cols if "(000)" in str(c) or "Weighted" in str(c)]
        
        if not default_data: 
            default_data = potential_data_cols[:5] # Fallback
            
        data_cols = st.sidebar.multiselect("Columns with Weighted Data:", potential_data_cols, default=default_data)
        
        if st.sidebar.button("Run Analysis") and data_cols:
            try:
                # PREPARE DATA
                cleaned_df = df.set_index(label_col)[data_cols]
                
                # Remove "Total" rows if present
                cleaned_df = cleaned_df[~cleaned_df.index.astype(str).str.contains("Total", case=False, na=False)]
                
                # Clean numbers (remove commas)
                for col in cleaned_df.columns:
                    cleaned_df[col] = cleaned_df[col].astype(str).str.replace(',', '', regex=False)
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                
                cleaned_df = cleaned_df.fillna(0)
                
                # RUN CA MATH
                N = cleaned_df.values
                if N.sum() == 0:
                    st.error("Selected data sums to zero. Please check your column selection.")
                    st.stop()
                    
                P = N / N.sum()
                r = P.sum(axis=1)
                c = P.sum(axis=0)
                E = np.outer(r, c)
                R = (P - E) / np.sqrt(E)
                U, s, Vh = np.linalg.svd(R, full_matrices=False)
                
                # STATS
                inertia = s**2
                explained = (inertia / np.sum(inertia)) * 100
                total_acc = explained[0] + explained[1]
                
                # VISUALIZE
                st.divider()
                col1, col2 = st.columns([1, 3])
                col1.metric("Map Accuracy", f"{total_acc:.1f}%")
                if total_acc < 60:
                    col2.error("⚠️ Unstable Map (<60% Accuracy)")
                else:
                    col2.success("✅ Stable Map")
                    
                # COORDS
                row_c = (U * s) / np.sqrt(r[:, np.newaxis])
                col_c = (Vh.T * s) / np.sqrt(c[:, np.newaxis])
                
                df_brands = pd.DataFrame(col_c[:, :2], columns=['x', 'y'])
                df_brands['Label'] = cleaned_df.columns
                df_brands['Type'] = 'Brand'
                
                df_attrs = pd.DataFrame(row_c[:, :2], columns=['x', 'y'])
                df_attrs['Label'] = cleaned_df.index
                df_attrs['Type'] = 'Attribute'
                
                plot_data = pd.concat([df_brands, df_attrs])
                
                fig = px.scatter(plot_data, x='x', y='y', text='Label', color='Type',
                                title="Perceptual Map", template="plotly_white", height=700,
                                color_discrete_map={'Brand': '#1f77b4', 'Attribute': '#d62728'})
                fig.update_traces(textposition='top center', marker_size=10)
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                fig.add_vline(x=0, line_dash="dot", line_color="gray")
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Math Error: {e}")
