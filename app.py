import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Universal Strategy Engine")

# --- MAIN APP ---
st.title("🧠 The Strategy Engine")
st.markdown("Upload your **Cleaned** CSV (Attributes in Col A, Brands in Cols B+).")
st.caption("💡 **Pro Tip:** Use 'Focus Mode' in the sidebar to tell a specific brand story.")

# 1. UPLOAD
uploaded_file = st.sidebar.file_uploader("Upload Clean CSV/Excel", type=["csv", "xlsx"])

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
df_brands = pd.DataFrame(col_coords[:, :2], columns=['x', 'y'])
df_brands['Label'] = cleaned_df.columns
df_brands['Type'] = 'Brand'
df_brands['Distinctiveness'] = np.sqrt(df_brands['x']**2 + df_brands['y']**2)

df_attrs = pd.DataFrame(row_coords[:, :2], columns=['x', 'y'])
df_attrs['Label'] = cleaned_df.index
df_attrs['Type'] = 'Attribute'
df_attrs['Distinctiveness'] = np.sqrt(df_attrs['x']**2 + df_attrs['y']**2)

# --- THE COMPASS: Find Axis Drivers ---
# We look for the attributes with the max/min X and Y values to label the axes
x_min_attr = df_attrs.loc[df_attrs['x'].idxmin()]['Label']
x_max_attr = df_attrs.loc[df_attrs['x'].idxmax()]['Label']
y_min_attr = df_attrs.loc[df_attrs['y'].idxmin()]['Label']
y_max_attr = df_attrs.loc[df_attrs['y'].idxmax()]['Label']

# --- UX CONTROLS ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Map Controls")

# Sliders
total_brands = len(df_brands)
n_brands_show = st.sidebar.slider("Brand Density", 2, total_brands, total_brands)

total_attrs = len(df_attrs)
n_attrs_show = st.sidebar.slider("Attribute Density", 5, total_attrs, 15)

show_attr_labels = st.sidebar.checkbox("Show Attribute Labels", value=True)

# Focus Mode
st.sidebar.markdown("---")
st.sidebar.subheader("🔦 Focus Mode")
focus_brand = st.sidebar.selectbox("Highlight a specific brand story:", ["None"] + df_brands['Label'].tolist())

# --- FILTERING LOGIC ---
top_brands = df_brands.sort_values('Distinctiveness', ascending=False).head(n_brands_show)
top_attrs = df_attrs.sort_values('Distinctiveness', ascending=False).head(n_attrs_show)

# Accuracy
inertia = s**2
accuracy = (np.sum(inertia[:2]) / np.sum(inertia)) * 100

st.divider()
col1, col2 = st.columns([1, 3])
col1.metric("Map Accuracy", f"{accuracy:.1f}%")

if focus_brand != "None":
    col2.info(f"🔦 **Focus Mode Active:** Highlighting **{focus_brand}** and its closest attributes.")
else:
    col2.caption(f"Showing **{n_brands_show}** brands and **{n_attrs_show}** attributes.")

# --- INTERACTIVE MAP ---
fig = go.Figure()

# 1. DEFINE COLORS based on Focus Mode
# Default Colors
brand_color_default = '#1f77b4'  # Professional Blue
attr_color_default = '#d62728'   # Professional Red
dim_color = '#d3d3d3'            # Light Grey for dimmed items

# Logic: If Focus Mode is ON, everything is Grey EXCEPT the hero brand
if focus_brand != "None":
    # Find the Hero Brand's position
    hero_brand = df_brands[df_brands['Label'] == focus_brand].iloc[0]
    
    # Calculate distance from Hero Brand to all visible attributes
    top_attrs['DistToHero'] = np.sqrt((top_attrs['x'] - hero_brand['x'])**2 + (top_attrs['y'] - hero_brand['y'])**2)
    
    # Identify top 5 related attributes
    hero_related_attrs = top_attrs.sort_values('DistToHero').head(5)['Label'].tolist()
    
    # Assign Colors
    brand_colors = [brand_color_default if x == focus_brand else dim_color for x in top_brands['Label']]
    attr_colors = [attr_color_default if x in hero_related_attrs else dim_color for x in top_attrs['Label']]
    
    # Assign Opacities (Hero pops out, others fade)
    brand_opacity = [1.0 if x == focus_brand else 0.3 for x in top_brands['Label']]
    attr_opacity = [1.0 if x in hero_related_attrs else 0.3 for x in top_attrs['Label']]
    
else:
    # Standard Mode
    brand_colors = brand_color_default
    attr_colors = attr_color_default
    brand_opacity = 1.0
    attr_opacity = 0.7

# 2. ADD TRACES
# Brands
fig.add_trace(go.Scatter(
    x=top_brands['x'], y=top_brands['y'],
    mode='markers',
    name='Brands',
    marker=dict(size=14, color=brand_colors, opacity=brand_opacity, line=dict(width=1, color='white')),
    text=top_brands['Label'],
    hovertemplate="<b>%{text}</b><br>Distinctiveness: %{customdata:.2f}<extra></extra>",
    customdata=top_brands['Distinctiveness']
))

# Attributes
fig.add_trace(go.Scatter(
    x=top_attrs['x'], y=top_attrs['y'],
    mode='markers',
    name='Attributes',
    marker=dict(size=8, color=attr_colors, opacity=attr_opacity),
    text=top_attrs['Label'],
    hovertemplate="<b>%{text}</b><extra></extra>"
))

# 3. DRAGGABLE LABELS
annotations = []

# Brand Labels
for i, row in top_brands.iterrows():
    # In Focus Mode, only label the Hero Brand
    if focus_brand != "None" and row['Label'] != focus_brand:
        continue
        
    annotations.append(dict(
        x=row['x'], y=row['y'],
        text=row['Label'],
        showarrow=True, arrowhead=0, arrowcolor=brand_color_default,
        ax=0, ay=-25,
        font=dict(size=13, color=brand_color_default, family="Arial Black"),
        bgcolor="rgba(255,255,255,0.8)"
    ))

# Attribute Labels
if show_attr_labels:
    for i, row in top_attrs.iterrows():
        # In Focus Mode, only label the related attributes
        if focus_brand != "None" and row['Label'] not in hero_related_attrs:
            continue
            
        annotations.append(dict(
            x=row['x'], y=row['y'],
            text=row['Label'],
            showarrow=True, arrowhead=0, arrowcolor=attr_color_default,
            ax=0, ay=-15,
            font=dict(size=10, color=attr_color_default),
            bgcolor="rgba(255,255,255,0.6)"
        ))

# 4. LAYOUT & THE COMPASS
fig.update_layout(
    annotations=annotations,
    title={
        'text': f"Strategic Map: {focus_brand}" if focus_brand != "None" else "Strategic Perceptual Map",
        'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
    },
    template="plotly_white",
    height=800,
    # The Compass: Auto-Labeling the Axes
    xaxis=dict(
        title=f"← More {x_min_attr} ........................... More {x_max_attr} →",
        title_font=dict(size=12, color='gray'),
        zeroline=True, zerolinewidth=2, zerolinecolor='gray', showgrid=False
    ),
    yaxis=dict(
        title=f"← More {y_min_attr} ... More {y_max_attr}
