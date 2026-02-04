import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Universal Strategy Engine")

# --- HELPER: ROBUST MRI LOADER ---
def load_data_advanced(uploaded_file, brand_row_num, metric_row_num):
    """
    Reads a file with a 'Split Header' using Excel Row Numbers (1-based).
    Merges headers and handles duplicate column names to prevent crashes.
    """
    try:
        # 1. Read the raw file with NO header
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, header=None, encoding='utf-8')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, header=None, encoding='latin1')
        else:
            df = pd.read_excel(uploaded_file, header=None)

        # 2. Convert Excel Row Numbers (1-based) to Pandas Index (0-based)
        brand_idx = int(brand_row_num) - 1
        metric_idx = int(metric_row_num) - 1

        # Safety Check
        if brand_idx >= len(df) or metric_idx >= len(df):
            return pd.DataFrame(), f"Row number out of bounds. File has only {len(df)} rows."

        # 3. Extract Rows
        # We take the raw row. We will force conversion in the loop.
        brand_row = df.iloc[brand_idx].values
        metric_row = df.iloc[metric_idx].values

        # 4. Forward Fill Brand Names (The Stitching)
        current_brand = ""
        combined_header = []
        
        for b, m in zip(brand_row, metric_row):
            # DEFENSIVE CODING: Force everything to string, handling NaNs
            b_str = str(b).strip()
            m_str = str(m).strip()
            
            # clean up 'nan' string artifacts if pandas converted NaN to 'nan'
            if b_str.lower() == 'nan': b_str = ""
            if m_str.lower() == 'nan': m_str = ""
            
            if b_str: 
                current_brand = b_str
            
            # Create Label: "Life Cereal | Weighted (000)"
            if current_brand and m_str:
                label = f"{current_brand} | {m_str}"
            elif current_brand:
                label = current_brand
            elif m_str:
                label = m_str
            else:
                label = "Unknown"
            
            combined_header.append(label)

        # 5. DEDUPLICATE COLUMN NAMES
        seen = {}
        unique_header = []
        for col in combined_header:
            if col in seen:
                seen[col] += 1
                unique_header.append(f"{col}.{seen[col]}")
            else:
                seen[col] = 0
                unique_header.append(col)

        # 6. Apply Header & Slice Data
        df_data = df.iloc[metric_idx + 1:].copy()
        df_data.columns = unique_header
        
        # 7. Final Cleanup
        df_data = df_data.dropna(how='all', axis=1).dropna(how='all', axis=0)
        
        return df_data, None

    except Exception as e:
        return pd.DataFrame(), str(e)

# --- MAIN APP UI ---
st.title("🧠 The Strategy Engine")
st.markdown("Upload your MRI/Simmons crosstab.")

# STEP 1: UPLOAD
uploaded_file = st.sidebar.file_uploader("Upload Crosstab", type=["csv", "xlsx"])

if uploaded_file is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Header Stitcher")
    
    # User Inputs for Row Numbers (EXCEL NUMBERS)
    st.sidebar.info("Enter the **Excel Row Numbers** exactly as you see them.")
    brand_row_num = st.sidebar.number_input("Row # with Brand Names", value=8, min_value=1)
    metric_row_num = st.sidebar.number_input("Row # with Metrics", value=11, min_value=1)
    
    # Load Data
    df, error = load_data_advanced(uploaded_file, brand_row_num, metric_row_num)
    
    if not df.empty:
        # Show result
        with st.expander("👀 Check Data (Success!)", expanded=True):
            st.write(f"Loaded {df.shape[0]} rows.")
            st.dataframe(df.head(3))
            
        st.sidebar.markdown("---")
        st.sidebar.subheader("🗺️ Map Configuration")

        # STEP 2: COLUMN MAPPING
        all_cols = df.columns.tolist()
        
        # Attribute Column
        label_col = st.sidebar.selectbox("Attribute Column (Rows):", all_cols, index=0)
        
        # Data Columns
        potential_cols = [c for c in all_cols if c != label_col]
        # Smart Select: Look for Weighted data
        default_cols = [c for c in potential_cols if "Weighted" in str(c) or "(000)" in str(c)]
        
        data_cols = st.sidebar.multiselect("Select Brand Columns:", potential_cols, default=default_cols)

        if st.sidebar.button("Run Analysis") and data_cols:
            try:
                # STEP 3: CLEANING
                cleaned_df = df.set_index(label_col)[data_cols]
                
                # Filter 'Total' rows
                cleaned_df = cleaned_df[~cleaned_df.index.astype(str).str.contains("Total", case=False, na=False)]
                
                # Clean Numbers
                for col in cleaned_df.columns:
                    cleaned_df[col] = cleaned_df[col].astype(str).str.replace(',', '', regex=False)
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                
                cleaned_df = cleaned_df.fillna(0)
                
                # Drop Zero Rows/Cols
                cleaned_df = cleaned_df.loc[(cleaned_df != 0).any(axis=1)]
                cleaned_df = cleaned_df.loc[:, (cleaned_df != 0).any(axis=0)]

                if cleaned_df.empty:
                    st.error("Data is empty after cleaning.")
                    st.stop()

                # STEP 4: MATH
                N = cleaned_df.values
                P = N / N.sum()
                r = P.sum(axis=1)
                c = P.sum(axis=0)
                E = np.outer(r, c)
                E[E < 1e-9] = 1e-9 # Safety
                R = (P - E) / np.sqrt(E)
                U, s, Vh = np.linalg.svd(R, full_matrices=False)
                
                # STEP 5: VISUALIZATION
                row_coords = (U * s) / np.sqrt(r[:, np.newaxis])
                col_coords = (Vh.T * s) / np.sqrt(c[:, np.newaxis])
                
                # Variance
                inertia = s**2
                accuracy = (np.sum(inertia[:2]) / np.sum(inertia)) * 100
                
                st.divider()
                st.metric("Map Accuracy", f"{accuracy:.1f}%")
                
                # Plot
                # Clean labels for plot (Remove '| Weighted')
                brand_labels = [c.split('|')[0].strip() for c in cleaned_df.columns]
                
                df_brands = pd.DataFrame(col_coords[:, :2], columns=['x', 'y'])
                df_brands['Label'] = brand_labels
                df_brands['Type'] = 'Brand'
                
                df_attrs = pd.DataFrame(row_coords[:, :2], columns=['x', 'y'])
                df_attrs['Label'] = cleaned_df.index
                df_attrs['Type'] = 'Attribute'
                
                plot_data = pd.concat([df_brands, df_attrs])
                
                fig = px.scatter(plot_data, x='x', y='y', text='Label', color='Type',
                                title="Strategic Perceptual Map", template="plotly_white", height=800,
                                color_discrete_map={'Brand': '#1f77b4', 'Attribute': '#d62728'})
                fig.update_traces(textposition='top center', marker_size=10)
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                fig.add_vline(x=0, line_dash="dot", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
                
                # White Space
                st.subheader("🔭 White Space Opportunities")
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
            st.error(f"Load Error: {error}")
