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
            
        st.sidebar.success("Data Loaded Successfully!")
        
        # 2. SELECT COLUMNS
        # Assume 1st column is labels, the rest are data
        label_col = df.columns[0] 
        all_brands = df.columns[1:].tolist()
        data_cols = st.sidebar.multiselect("Select Brands to Map", all_brands, default=all_brands)
        
        if st.sidebar.button("Run Analysis") and data_cols:
            
            # 3. PREPARE DATA
            cleaned_df = df.set_index(label_col)[data_cols]
            
            # Clean commas & convert to numbers (e.g. "1,200" -> 1200)
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce')
            
            cleaned_df = cleaned_df.fillna(0)
            
            # Remove empty rows/cols (Safety Check)
            cleaned_df = cleaned_df.loc[(cleaned_df != 0).any(axis=1)]
            cleaned_df = cleaned_df.loc[:, (cleaned_df != 0).any(axis=0)]
            
            if cleaned_df.empty:
                st.error("Error: Data is empty. Check that your CSV has numbers.")
                st.stop()

            # 4. THE MATH (SVD)
            N = cleaned_df.values
            P = N / N.sum()
            r = P.sum(axis=1)
            c = P.sum(axis=0)
            E = np.outer(r, c)
            E[E < 1e-9] = 1e-9 # Safety buffer
            R = (P - E) / np.sqrt(E)
            U, s, Vh = np.linalg.svd(R, full_matrices=False)
            
            # 5. VISUALIZATION CONFIG
            row_coords = (U * s) / np.sqrt(r[:, np.newaxis])
            col_coords = (Vh.T * s) / np.sqrt(c[:, np.newaxis])
            
            # Brands Dataframe
            df_brands = pd.DataFrame(col_coords[:, :2], columns=['x', 'y'])
            df_brands['Label'] = cleaned_df.columns
            df_brands['Type'] = 'Brand'
            
            # Attributes Dataframe
            df_attrs = pd.DataFrame(row_coords[:, :2], columns=['x', 'y'])
            df_attrs['Label'] = cleaned_df.index
            df_attrs['Type'] = 'Attribute'
            
            plot_data = pd.concat([df_brands, df_attrs])
            
            # --- GOLDILOCKS ROUNDING (2 Decimals) ---
            plot_data['x'] = plot_data['x'].round(2)
            plot_data['y'] = plot_data['y'].round(2)
            
            # Calculate Accuracy
            inertia = s**2
            accuracy = (np.sum(inertia[:2]) / np.sum(inertia)) * 100
            
            st.divider()
            col1, col2 = st.columns([1, 3])
            col1.metric("Map Accuracy", f"{accuracy:.1f}%")
            if accuracy < 60:
                col2.warning("⚠️ Low Accuracy (< 60%). The map may be misleading.")
            else:
                col2.success("✅ High Accuracy. The map is reliable.")
            
            # Plot
            fig = px.scatter(plot_data, x='x', y='y', text='Label', color='Type',
                            title="Strategic Perceptual Map", template="plotly_white", height=750,
                            color_discrete_map={'Brand': '#1f77b4', 'Attribute': '#d62728'},
                            hover_data={'x':':.2f', 'y':':.2f'}) # Tooltip format .2f
            
            fig.update_traces(textposition='top center', marker_size=12)
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            fig.add_vline(x=0, line_dash="dot", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
            
            # 6. WHITE SPACE FINDER
            st.subheader("🔭 Opportunity Finder")
            
            # Calculate raw isolation scores (Distance to nearest brand)
            scores = []
            for _, attr in df_attrs.iterrows():
                dists = np.linalg.norm(df_brands[['x','y']].values - attr[['x','y']].values, axis=1)
                scores.append(dists.min())
            
            df_attrs['Isolation'] = scores
            # Round scores to 2 decimals for display
            df_attrs['Isolation'] = df_attrs['Isolation'].round(2)
            
            top_opps = df_attrs.sort_values('Isolation', ascending=False).head(5)[['Label', 'Isolation']]
            st.table(top_opps)
            
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.info("Tip: Make sure your CSV has just one header row!")
