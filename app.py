import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Universal Strategy Engine")

# --- MAIN APP ---
st.title("🧠 The Strategy Engine")
st.markdown("Upload your **Cleaned** CSV (Attributes in Col A, Brands in Cols B+).")
st.caption("💡 **Pro Tip:** Scroll down to see the AI-generated story of the map.")

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

# --- THE COMPASS: FINDING THEMES ---
# We grab the top 2 attributes for each cardinal direction to create a "Theme"
def get_theme(df, direction):
    if direction == 'Right': # Max X
        return " & ".join(df.sort_values('x', ascending=False).head(2)['Label'].tolist())
    elif direction == 'Left': # Min X
        return " & ".join(df.sort_values('x', ascending=True).head(2)['Label'].tolist())
    elif direction == 'Top': # Max Y
        return " & ".join(df.sort_values('y', ascending=False).head(2)['Label'].tolist())
    elif direction == 'Bottom': # Min Y
        return " & ".join(df.sort_values('y', ascending=True).head(2)['Label'].tolist())

theme_right = get_theme(df_attrs, 'Right')
theme_left = get_theme(df_attrs, 'Left')
theme_top = get_theme(df_attrs, 'Top')
theme_bottom = get_theme(df_attrs, 'Bottom')

# --- UX CONTROLS ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Map Controls")
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
inertia = s**2
accuracy = (np.sum(inertia[:2]) / np.sum(inertia)) * 100

st.divider()
col1, col2 = st.columns([1, 3])
col1.metric("Map Accuracy", f"{accuracy:.1f}%")

if focus_brand != "None":
    col2.info(f"🔦 **Focus Mode Active:** Highlighting **{focus_brand}**.")
else:
    col2.caption(f"Showing **{n_brands_show}** brands and **{n_attrs_show}** attributes.")

# --- INTERACTIVE MAP ---
fig = go.Figure()

# 1. DEFINE COLORS
brand_color_default = '#1f77b4'
attr_color_default = '#d62728'
dim_color = '#d3d3d3'

if focus_brand != "None":
    hero_brand = df_brands[df_brands['Label'] == focus_brand].iloc[0]
    top_attrs['DistToHero'] = np.sqrt((top_attrs['x'] - hero_brand['x'])**2 + (top_attrs['y'] - hero_brand['y'])**2)
    hero_related_attrs = top_attrs.sort_values('DistToHero').head(5)['Label'].tolist()
    
    brand_colors = [brand_color_default if x == focus_brand else dim_color for x in top_brands['Label']]
    attr_colors = [attr_color_default if x in hero_related_attrs else dim_color for x in top_attrs['Label']]
    brand_opacity = [1.0 if x == focus_brand else 0.3 for x in top_brands['Label']]
    attr_opacity = [1.0 if x in hero_related_attrs else 0.3 for x in top_attrs['Label']]
else:
    brand_colors = brand_color_default
    attr_colors = attr_color_default
    brand_opacity = 1.0
    attr_opacity = 0.7

# 2. ADD TRACES
fig.add_trace(go.Scatter(
    x=top_brands['x'], y=top_brands['y'],
    mode='markers',
    name='Brands',
    marker=dict(size=14, color=brand_colors, opacity=brand_opacity, line=dict(width=1, color='white')),
    text=top_brands['Label'],
    hovertemplate="<b>%{text}</b><br>Distinctiveness: %{customdata:.2f}<extra></extra>",
    customdata=top_brands['Distinctiveness']
))

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
for i, row in top_brands.iterrows():
    if focus_brand != "None" and row['Label'] != focus_brand: continue
    annotations.append(dict(
        x=row['x'], y=row['y'],
        text=row['Label'],
        showarrow=True, arrowhead=0, arrowcolor=brand_color_default,
        ax=0, ay=-25,
        font=dict(size=13, color=brand_color_default, family="Arial Black"),
        bgcolor="rgba(255,255,255,0.8)"
    ))

if show_attr_labels:
    for i, row in top_attrs.iterrows():
        if focus_brand != "None" and row['Label'] not in hero_related_attrs: continue
        annotations.append(dict(
            x=row['x'], y=row['y'],
            text=row['Label'],
            showarrow=True, arrowhead=0, arrowcolor=attr_color_default,
            ax=0, ay=-15,
            font=dict(size=10, color=attr_color_default),
            bgcolor="rgba(255,255,255,0.6)"
        ))

# 4. LAYOUT & STORY AXES
fig.update_layout(
    annotations=annotations,
    title={
        'text': f"Strategic Map: {focus_brand}" if focus_brand != "None" else "Strategic Perceptual Map",
        'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
    },
    template="plotly_white",
    height=800,
    # The Story Axes (Top 2 Attributes)
    xaxis=dict(
        title=f"← {theme_left} ........................................... {theme_right} →",
        title_font=dict(size=14, color='black'),
        zeroline=True, zerolinewidth=2, zerolinecolor='gray', showgrid=False
    ),
    yaxis=dict(
        title=f"← {theme_bottom} ... {theme_top} →",
        title_font=dict(size=14, color='black'),
        zeroline=True, zerolinewidth=2, zerolinecolor='gray', showgrid=False
    ),
    showlegend=False,
    dragmode='pan'
)

# Quadrant Backgrounds
fig.add_shape(type="rect", x0=0, y0=0, x1=df_brands['x'].max()*1.2, y1=df_brands['y'].max()*1.2, 
              fillcolor="blue", opacity=0.03, layer="below", line_width=0)
fig.add_shape(type="rect", x0=df_brands['x'].min()*1.2, y0=df_brands['y'].min()*1.2, x1=0, y1=0, 
              fillcolor="blue", opacity=0.03, layer="below", line_width=0)

st.plotly_chart(fig, use_container_width=True, config={'editable': True, 'scrollZoom': True})

# --- AUTOMATED STORY GENERATOR ---
st.subheader("📝 Map Interpretation")

# Find 'Winner' brands for each quadrant
def get_quadrant_winner(df, q_x, q_y):
    # Filter for quadrant (e.g., x > 0 and y > 0)
    quad = df[(df['x'] * q_x > 0) & (df['y'] * q_y > 0)]
    if not quad.empty:
        return quad.sort_values('Distinctiveness', ascending=False).iloc[0]['Label']
    return "No clear leader"

winner_ne = get_quadrant_winner(df_brands, 1, 1)   # Top Right
winner_nw = get_quadrant_winner(df_brands, -1, 1)  # Top Left
winner_se = get_quadrant_winner(df_brands, 1, -1)  # Bottom Right
winner_sw = get_quadrant_winner(df_brands, -1, -1) # Bottom Left

explanation = f"""
**The Main Conflict (Horizontal):** The biggest difference in this market is between **{theme_left}** on the left and **{theme_right}** on the right. 
This dimension explains the majority of the strategic variance.

**The Secondary Conflict (Vertical):** Brands also split based on being more **{theme_top}** (Top) versus **{theme_bottom}** (Bottom).

**The Strategic Territories (Quadrants):**
* **Top Right ({theme_right} & {theme_top}):** Dominated by **{winner_ne}**.
* **Top Left ({theme_left} & {theme_top}):** Dominated by **{winner_nw}**.
* **Bottom Right ({theme_right} & {theme_bottom}):** Dominated by **{winner_se}**.
* **Bottom Left ({theme_left} & {theme_bottom}):** Dominated by **{winner_sw}**.
"""

st.markdown(explanation)
