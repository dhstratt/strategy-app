import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Universal Strategy Engine")

# --- MAIN APP ---
st.title("🧠 The Strategy Engine")
st.markdown("Upload your **Cleaned** CSV (Attributes in Col A, Brands in Cols B+).")
st.caption("💡 **Pro Tip:** You can CLICK and DRAG labels on the map to organize them! (Snapshot before refreshing)")

# 1. UPLOAD
uploaded_file = st.sidebar.file_uploader("Upload Clean CSV/Excel", type=["csv", "xlsx"])

# SAFETY CHECK: Stop here if no file is uploaded (Prevents indentation errors)
if uploaded_file is None:
    st.info("👈 Waiting for file upload...")
    st.stop()

# 2. LOAD DATA
if uploaded_file.name.endswith('.csv'):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

st.sidebar.success("Data Loaded!")

# 3. SELECT COLUMNS
label_col = df.columns[0] 
all_brands = df.columns[1:].tolist()
data_cols = st.sidebar.multiselect("Select Brands to Map", all_brands, default=all_brands)

# SAFETY CHECK: Stop if no brands selected
if not data_cols:
    st.warning("Please select at least one brand.")
    st.stop()

# 4. PREPARE DATA
cleaned_df = df.set_index(label_col)[data_cols]

# Clean commas/numbers
for col in cleaned_df.columns:
    if cleaned_df[col].dtype == 'object':
        cleaned_df[col] = cleaned_df[col].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce')

cleaned_df = cleaned_df.fillna(0)
cleaned_df = cleaned_df.loc[(cleaned_df != 0).any(axis=1)]
cleaned_df = cleaned_df.loc[:, (cleaned_df != 0).any(axis=0)]

if cleaned_df.empty:
    st.error("Error: Data is empty after cleaning.")
    st.stop()

# 5. THE MATH (SVD)
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

# 6. VISUALIZATION DATAFRAMES
# Brands
df_brands = pd.DataFrame(col_coords[:, :2], columns=['x', 'y'])
df_brands['Label'] = cleaned_df.columns
df_brands['Type'] = 'Brand'
df_brands['Size'] = 12  

# Attributes
df_attrs = pd.DataFrame(row_coords[:, :2], columns=['x', 'y'])
df_attrs['Label'] = cleaned_df.index
df_attrs['Type'] = 'Attribute'
df_attrs['Size'] = 6   

# Strategic Importance
df_attrs['Importance'] = np.sqrt(df_attrs['x']**2 + df_attrs['y']**2)

# --- UX CONTROLS ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Design Controls")

total_attrs = len(df_attrs)
n_show = st.sidebar.slider("Attribute Density", min_value=5, max_value=total_attrs, value=15)
show_attr_labels = st.sidebar.checkbox("Show Attribute Labels", value=True)

# Filter Attributes
top_attrs = df_attrs.sort_values('Importance', ascending=False).head(n_show)

# Accuracy
inertia = s**2
accuracy = (np.sum(inertia[:2]) / np.sum(inertia)) * 100

st.divider()
col1, col2 = st.columns([1, 3])
col1.metric("Map Accuracy", f"{accuracy:.1f}%")
col2.caption(f"Showing top {n_show} attributes. **Drag labels to organize.**")

# --- INTERACTIVE MAP ---
fig = go.Figure()

# Add Brands (Dots)
fig.add_trace(go.Scatter(
    x=df_brands['x'], y=df_brands['y'],
    mode='markers', 
    name='Brands',
    marker=dict(size=12, color='#1f77b4', line=dict(width=1, color='white')),
    hoverinfo='text',
    hovertext=df_brands['Label']
))

# Add Attributes (Dots)
fig.add_trace(go.Scatter(
    x=top_attrs['x'], y=top_attrs['y'],
    mode='markers',
    name='Attributes',
    marker=dict(size=7, color='#d62728', opacity=0.7),
    hoverinfo='text',
    hovertext=top_attrs['Label']
))

# Add Draggable Annotations
annotations = []

# Brand Labels 
for i, row in df_brands.iterrows():
    annotations.append(dict(
        x=row['x'], y=row['y'],
        text=row['Label'],
        showarrow=True,       
        arrowhead=0,          
        arrowcolor="#1f77b4",
        ax=0, ay=-25,         
        font=dict(size=14, color="#1f77b4", family="Arial Black"), 
        bgcolor="rgba(255,255,255,0.8)" 
    ))

# Attribute Labels 
if show_attr_labels:
    for i, row in top_attrs.iterrows():
        annotations.append(dict(
            x=row['x'], y=row['y'],
            text=row['Label'],
            showarrow=True,
            arrowhead=0,
            arrowcolor="#d62728",
            ax=0, ay=-15,
            font=dict(size=11, color="#d62728"), 
            bgcolor="rgba(255,255,255,0.6)"
        ))

fig.update_layout(
    annotations=annotations,
    title="Strategic Perceptual Map (Interactive Designer)",
    template="plotly_white",
    height=800,
    xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray', showgrid=False),
    yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray', showgrid=False),
    showlegend=True,
    dragmode='pan' 
)

# Enable dragging
st.plotly_chart(fig, use_container_width=True, config={'editable': True, 'scrollZoom': True})
