import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2_contingency

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Universal Strategy Engine")

# --- HELPER FUNCTION: THE SMART LOADER ---
def load_mri_style_crosstab(uploaded_file):
    """
    Intelligently parses MRI-Simmons style crosstabs.
    Scans for the header row and cleans metadata.
    """
    # 1. Read the raw file to find the header
    if uploaded_file.name.endswith('.csv'):
        # Read first 20 rows to inspect
        raw_preview = pd.read_csv(uploaded_file, header=None, nrows=20)
    else:
        raw_preview = pd.read_excel(uploaded_file, header=None, nrows=20)
        
    # 2. Find the row that contains 'Total' or typical brand headers
    header_row_idx = 0
    for i, row in raw_preview.iterrows():
        # Heuristic: Look for a row that has more than 5 non-null values 
        # OR contains specific keywords like "Total" or "Weighted"
        if row.count() > 5:
            header_row_idx = i
            break
            
    # 3. Reload with the correct header
    if uploaded_file.name.endswith('.csv'):
        uploaded_file.seek(0) # Reset file pointer
        df = pd.read_csv(uploaded_file, header=header_row_idx)
    else:
        df = pd.read_excel(uploaded_file, header=header_row_idx)

    # 4. Cleanup: Remove empty columns and rows
    df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)
    
    # 5. Filter out metadata rows at the bottom (usually contain copyright info)
    # We assume the first column is the "Question/Attribute" column
    first_col = df.columns[0]
    df = df[~df[first_col].astype(str).str.contains("Exported from|MRI-Simmons|Copyright", case=False, na=False)]
    
    return df

# --- MAIN APP UI ---
st.title("🧠 The Strategy Engine: Universal Map Builder")
st.markdown("Upload a crosstab (Brands x Attributes) to generate a Perceptual Map.")

# STEP 1: UPLOAD
uploaded_file = st.sidebar.file_uploader("Upload Crosstab (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Use Smart Loader
        df = load_mri_style_crosstab(uploaded_file)
        
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
        st.sidebar.markdown("---")
        
        # STEP 2: COLUMN SELECTION
        # We need to distinguish between the Attribute Labels (Rows) and Brand Data (Columns)
        all_cols = df.columns.tolist()
        
        # Ask user for the 'Row Labels' column (e.g., 'Cereal Attributes')
        label_col = st.sidebar.selectbox("1. Which column has the Statement/Attribute text?", all_cols, index=0)
        
        # Ask user to select Brand Data Columns
        # Heuristic: Pre-select columns that look like numeric data or contain "(000)"
        potential_data_cols = [c for c in all_cols if c != label_col]
        default_selection = [c for c in potential_data_cols if "(000)" in str(c) or "Weighted" in str(c)]
        
        # If heuristics fail, just select the first 5
        if not default_selection:
            default_selection = potential_data_cols[:5]
            
        data_cols = st.sidebar.multiselect("2. Select Brand Columns (Weighted Data)", potential_data_cols, default=default_selection)
        
        if st.sidebar.button("Generate Map") and data_cols:
            
            # STEP 3: PREPARE DATA FOR MATH
            # Clean non-numeric data (remove commas, convert to float)
            cleaned_df = df.set_index(label_col)[data_cols]
            
            # Remove "Total" rows/columns if they exist (they distort the map)
            cleaned_df = cleaned_df[~cleaned_df.index.astype(str).str.contains("Total", case=False, na=False)]
            
            # Convert to numeric, forcing errors to NaN then filling with 0
            # This handles comma-formatted numbers like "1,200"
            for col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].astype(str).str.replace(',', '', regex=False)
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            
            cleaned_df = cleaned_df.fillna(0)
            
            # STEP 4: THE MATH (Correspondence Analysis)
            # CA Calculation Engine
            N = cleaned_df.values
            if N.sum() == 0:
                st.error("Data sums to zero. Check your column selection.")
                st.stop()
                
            P = N / N.sum()
            r = P.sum(axis=1)
            c = P.sum(axis=0)
            E = np.outer(r, c)
            R = (P - E) / np.sqrt(E)
            U, s, Vh = np.linalg.svd(R, full_matrices=False)
            
            # Explained Variance (Stability Check)
            inertia = s**2
            total_inertia = np.sum(inertia)
            explained_var = (inertia / total_inertia) * 100
            map_accuracy = explained_var[0] + explained_var[1]

            # STEP 5: VISUALIZATION
            st.divider()
            
            # Stability Alert
            col_kpi, col_msg = st.columns([1, 3])
            col_kpi.metric("Map Accuracy", f"{map_accuracy:.1f}%")
            if map_accuracy < 60:
                col_msg.error("⚠️ Map is Unstable (Low Variance Explained). Use with caution.")
            else:
                col_msg.success("✅ Map is Stable. Reliable for strategy.")

            # Calculate Coordinates
            row_coords = (U * s) / np.sqrt(row_sums[:, np.newaxis] if 'row_sums' in locals() else r[:, np.newaxis])
            col_coords = (Vh.T * s) / np.sqrt(col_sums[:, np.newaxis] if 'col_sums' in locals() else c[:, np.newaxis])

            # Create Plot Data
            # Brands
            map_data_brands = pd.DataFrame(col_coords[:, :2], columns=['x', 'y'])
            map_data_brands['Label'] = cleaned_df.columns
            map_data_brands['Type'] = 'Brand'
            
            # Attributes
            map_data_attrs = pd.DataFrame(row_coords[:, :2], columns=['x', 'y'])
            map_data_attrs['Label'] = cleaned_df.index
            map_data_attrs['Type'] = 'Attribute'
            
            full_map_data = pd.concat([map_data_brands, map_data_attrs])

            # Plotly Chart
            fig = px.scatter(
                full_map_data, x='x', y='y', text='Label', color='Type',
                title="Strategic Perceptual Map",
                color_discrete_map={'Brand': '#1f77b4', 'Attribute': '#d62728'},
                height=700,
                template="plotly_white"
            )
            fig.update_traces(textposition='top center', marker_size=10)
            fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
            fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
            fig.update_layout(showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # STEP 6: WHITE SPACE FINDER (Bonus)
            st.subheader("🔭 White Space Finder")
            # Simple logic: Find attributes furthest from any brand
            iso_scores = []
            for _, attr in map_data_attrs.iterrows():
                # Distance to nearest brand
                dists = np.linalg.norm(map_data_brands[['x', 'y']].values - attr[['x', 'y']].values, axis=1)
                iso_scores.append(dists.min())
            
            map_data_attrs['Isolation'] = iso_scores
            top_opps = map_data_attrs.sort_values('Isolation', ascending=False).head(5)
            
            st.write("Top Unclaimed Opportunities:")
            st.dataframe(top_opps[['Label', 'Isolation']])

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Tip: Make sure you selected numeric columns for the Brand Data.")
