import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Universal Strategy Engine")

# --- HELPER: ADVANCED MRI LOADER ---
def load_data_advanced(uploaded_file, brand_row_idx, metric_row_idx):
    """
    Reads a file with a 'Split Header' (Brand names on one row, Metrics on another).
    Merges them into a single descriptive header.
    """
    try:
        # 1. Read the raw file with NO header so we can grab specific rows
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, header=None, encoding='utf-8')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, header=None, encoding='latin1')
        else:
            df = pd.read_excel(uploaded_file, header=None)

        # 2. Extract the two header rows
        # Subtract 1 because pandas is 0-indexed (Row 8 is index 7)
        # But we will use the user's input directly if they use 0-based counting, 
        # or adjust if they use 1-based. Let's assume 0-based for code, but labels helper.
        
        # We'll use the inputs directly as 0-indexed for simplicity in logic
        # Safety check
        if brand_row_idx >= len(df) or metric_row_idx >= len(df):
            return pd.DataFrame(), "Row index out of bounds"

        brand_row = df.iloc[brand_row_idx].copy()
        metric_row = df.iloc[metric_row_idx].astype(str).copy()

        # 3. THE FIX: Forward Fill the Brand Names
        # If Col 1 is "Life" and Col 2 is NaN, Col 2 becomes "Life"
        brand_row = brand_row.ffill()
        
        # 4. Clean "nan" brands (columns that had no brand above them)
        brand_row = brand_row.fillna("")

        # 5. Combine them: "Life Cereal | Weighted (000)"
        combined_header = []
        for b, m in zip(brand_row, metric_row):
            clean_b = str(b).strip()
            clean_m = str(m).strip()
            if clean_b and clean_m:
                combined_header.append(f"{clean_b} | {clean_m}")
            elif clean_b:
                combined_header.append(clean_b)
            else:
                combined_header.append(clean_m)

        # 6. Apply new header and slice data
        # Data starts immediately after the metric row
        df_data = df.iloc[metric_row_idx + 1:].copy()
        df_data.columns = combined_header
        
        # 7. Reset index and Drop empty rows
        df_data = df_data.dropna(how='all', axis=1).dropna(how='all', axis=0)
        
        return df_data, None

    except Exception as e:
        return pd.DataFrame(), str(e)

# --- MAIN APP UI ---
st.title("🧠 The Strategy Engine")
st.markdown("Upload your MRI/Simmons crosstab. We will stitch the headers for you.")

# STEP 1: UPLOAD
uploaded_file = st.sidebar.file_uploader("Upload Crosstab", type=["csv", "xlsx"])

if uploaded_file is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Header Stitcher")
    st.sidebar.info("Look at your Excel file. Which row has the **Brand Names** and which has the **Metrics**?")
    
    # User Inputs for Row Numbers
    # Defaulting to 8 and 11 based on your file 'Test 1'
    brand_row_num = st.sidebar.number_input("Row # with Brand Names (e.g., 8)", value=8, min_value=0)
    metric_row_num = st.sidebar.number_input("Row # with Metrics (e.g., 11)", value=11, min_value=0)
    
    # Load Data
    df, error = load_data_advanced(uploaded_file, brand_row_num, metric_row_num)
    
    if not df.empty:
        # Show the user the result
        with st.expander("👀 Check Data (Columns should now include Brand Names)", expanded=True):
            st.write(f"Loaded {df.shape[0]} rows. First 3 rows:")
            st.dataframe(df.head(3))
            
        st.sidebar.markdown("---")
        st.sidebar.subheader("🗺️ Map Your Data")

        # STEP 2: COLUMN MAPPING
        all_cols = df.columns.tolist()
        
        # 1. Attribute Column (usually the first one)
        label_col = st.sidebar.selectbox("Attribute Column:", all_cols, index=0)
        
        # 2. Data Columns (Filter for "Weighted")
        # Now we search for columns that contain BOTH the brand and "Weighted"
        potential_cols = [c for c in all_cols if c != label_col]
        
        # Smart Select: Look for "Weighted" or "(000)"
        default_cols = [c for c in potential_cols if "Weighted" in str(c) or "(000)" in str(c)]
        
        # If multiple metrics exist, user might see "Life | Weighted" and "Life | Vert%"
        data_cols = st.sidebar.multiselect("Select 'Weighted' Columns for Brands:", potential_cols, default=default_cols)

        if st.sidebar.button("Run Analysis") and data_cols:
            try:
                # STEP 3: CLEANING
                cleaned_df = df.set_index(label_col)[data_cols]
                
                # Remove "Total" rows
                cleaned_df = cleaned_df[~cleaned_df.index.astype(str).str.contains("Total", case=False, na=False)]
                
                # Clean Numbers
                for col in cleaned_df.columns:
                    cleaned_df[col] = cleaned_df[col].astype(str).str.replace(',', '', regex=False)
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                
                cleaned_df = cleaned_df.fillna(0)
                
                # Drop Zero Rows/Cols
                cleaned_df = cleaned_df.loc[(cleaned_df != 0).any(axis=1)] # Rows
                cleaned_df = cleaned_df.loc[:, (cleaned_df != 0).any(axis=0)] # Cols

                if cleaned_df.empty:
                    st.error("Data is empty after cleaning. Check your column selection.")
                    st.stop()

                # STEP 4: MATH ENGINE
                N = cleaned_df.values
                P = N / N.sum()
                r = P.sum(axis=1)
                c = P.sum(axis=0)
                E = np.outer(r, c)
                
                # Safety Buffer
                E[E < 1e-9] = 1e-9
                
                R = (P - E) / np.sqrt(E)
                U, s, Vh = np.linalg.svd(R, full_matrices=False)
                
                # STEP 5: VISUALIZATION
                row_coords = (U * s) / np.sqrt(r[:, np.newaxis])
                col_coords = (Vh.T * s) / np.sqrt(c[:, np.newaxis])
                
                # Accuracy
                inertia = s**2
                accuracy = (np.sum(inertia[:2]) / np.sum(inertia)) * 100
                
                st.divider()
                st.metric("Map Accuracy", f"{accuracy:.1f}%")
                
                # Prepare Plot
                # Use Cleaned Names for labels (Remove the "| Weighted" part for cleaner chart)
                brand_labels = [c.split('|')[0].strip() for c in cleaned_df.columns]
                
                df_brands = pd.DataFrame(col_coords[:, :2], columns=['x', 'y'])
                df_brands['Label'] = brand_labels
                df_brands['Type'] = 'Brand'
                
                df_attrs = pd.DataFrame(row_coords[:, :2], columns=['x', 'y'])
                df_attrs['Label'] = cleaned_df.index
                df_attrs['Type'] = 'Attribute'
                
                plot_data = pd.concat([df_brands, df_attrs])
                
                fig = px.scatter(plot_data, x='x', y='y', text='Label', color='Type',
                                title="Strategic Perceptual Map", template="plotly_white", height=700,
                                color_discrete_map={'Brand': '#1f77b4', 'Attribute': '#d62728'})
                fig.update_traces(textposition='top center', marker_size=10)
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                fig.add_vline(x=0, line_dash="dot", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
                
                # White Space Finder
                st.subheader("🔭 White Space Opportunities")
                # Calculate simple distance to nearest brand
                scores = []
                for _, attr in df_attrs.iterrows():
                    dists = np.linalg.norm(df_brands[['x','y']].values - attr[['x','y']].values, axis=1)
                    scores.append(dists.min())
                
                df_attrs['Isolation'] = scores
                st.dataframe(df_attrs.sort_values('Isolation', ascending=False).head(5)[['Label', 'Isolation']])

            except Exception as e:
                st.error(f"Analysis Error: {e}")
    else:
        if error:
            st.error(f"Could not load data: {error}")
