import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import collections

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Universal Strategy Engine")

# --- MAIN APP ---
st.title("🧠 The Strategy Engine")
st.markdown("Upload your **Cleaned** CSV (Attributes in Col A, Brands in Cols B+).")
st.caption("💡 **Pro Tip:** Check the sidebar to rename the axes and 'interpret' the map strategy.")

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

# --- THE INTERPRETATION ENGINE ---
# This function finds the most common meaningful word in the top attributes to "Guess" the theme
def suggest_theme(df, direction, n=4):
    # Sort by direction
    if direction == 'Right': subset = df.sort_values('x', ascending=False).head(n)['Label']
    elif direction == 'Left': subset = df.sort_values('x', ascending=True).head(n)['Label']
    elif direction == 'Top': subset = df.sort_values('y', ascending=False).head(n)['Label']
    elif direction == 'Bottom': subset = df.sort_values('y', ascending=True).head(n)['Label']
    
    # 1. Simple Join (Backup)
    top_1 = subset.iloc[0]
    
    # 2. Smart Word Extraction
    all_text = " ".join(subset.astype(str)).lower()
    # Remove survey filler words
    stop_words = ['i', 'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'my', 'for', 'on', 'with', 'often', 'prefer', 'like', 'typically', 'look', 'buy', 'make', 'feel', 'more', 'less', 'most']
    words = [w.strip(".,()&") for w in all_text.split() if w.strip(".,()&") not in stop_words and len(w) > 2]
    
    if words:
        most_common = collections.Counter(words).most_common(1)[0]
        # If the most common word appears more than once, use it as the theme
        if most_common[1] > 1:
            return most_common[0].capitalize() # e.g., "Healthy"
            
    # Fallback: Return the top attribute (truncated)
    return (top_1[:30] + '..') if len(top_1) > 30 else top_1

# Generate Suggestions
sugg_right = suggest_theme(df_attrs, 'Right')
sugg_left = suggest_theme(df_attrs, 'Left')
sugg_top = suggest_theme(df_attrs, 'Top')
sugg_bottom = suggest_theme(df_attrs, 'Bottom')

# --- SIDEBAR: STRATEGIST CONTROLS ---
st.sidebar.markdown("---")
st.sidebar.subheader("🏷️ Axis Interpretation")
st.sidebar.caption("The app has guessed the themes based on attribute clusters. **Rename them below to interpret the strategy.**")

# Editable Inputs
theme_left = st.sidebar.text_input("← Left Axis Theme", value=sugg_left)
theme_right = st.sidebar.text_input("Right Axis Theme →", value=sugg_right)
theme_bottom = st.sidebar.text_input("↓ Bottom Axis Theme", value=sugg_bottom)
theme_top = st.sidebar.text_input("Top Axis Theme ↑", value=sugg_top)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Display Controls")
total_brands = len(df_brands)
n_brands_show = st.sidebar.slider("Brand Density", 2, total_brands, total_brands)
total_attrs = len(df_attrs)
n_attrs_show = st.sidebar.slider("Attribute Density", 5, total_attrs, 15)
show_attr_labels = st.sidebar.checkbox("Show Attribute Labels", value=True)

# Focus Mode
st.sidebar.markdown("---")
st.sidebar.subheader("🔦 Focus Mode")
focus_brand = st.sidebar.selectbox("Highlight a specific brand:", ["None"] + df_brands['Label'].tolist())

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
    col2.caption(f"Map interprets: **{theme_left} vs. {theme_right}** and **{theme_top} vs. {theme_bottom}**.")

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

# 4. LAYOUT & INTERPRETATION AXES
fig.update_layout(
    annotations=annotations,
    title={
        'text': f"Strategic Map: {focus_brand}" if focus_brand != "None" else "Strategic Perceptual Map",
        'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
    },
    template="plotly_white",
    height=800,
    # The Interpreted Axes
    xaxis=dict(
        title=f"← {theme_left} ........................................... {theme_right} →",
        title_font=dict(size=16, color='black', family="Arial Black"),
        zeroline=True, zerolinewidth=2, zerolinecolor='gray', showgrid=False
    ),
    yaxis=dict(
        title=f"← {theme_bottom} ... {theme_top} →",
        title_font=dict(size=16, color='black', family="Arial Black"),
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

# --- AUTOMATED STORY GENERATOR (Uses Your Labels) ---
st.subheader("📝 Strategic Interpretation")

# Find 'Winner' brands for each quadrant
def get_quadrant_winner(df, q_x, q_y):
    quad = df[(df['x'] * q_x > 0) & (df['y'] * q_y > 0)]
    if not quad.empty:
        return quad.sort_values('Distinctiveness', ascending=False).iloc[0]['Label']
    return "Niche Players"

winner_ne = get_quadrant_winner(df_brands, 1, 1)   # Top Right
winner_nw = get_quadrant_winner(df_brands, -1, 1)  # Top Left
winner_se = get_quadrant_winner(df_brands, 1, -1)  # Bottom Right
winner_sw = get_quadrant_winner(df_brands, -1, -1) # Bottom Left

# Story Logic
explanation = f"""
**1. The Primary Tension (Horizontal):** The market is primarily divided by **{theme_left}** vs. **{theme_right}**. This is the single biggest differentiator between brands in this category.

**2. The Secondary Tension (Vertical):** There is a secondary split driven by **{theme_top}** vs. **{theme_bottom}**.

**3. The Strategic Battlegrounds:**
* **The "High {theme_right} / High {theme_top}" Territory:** Currently dominated by **{winner_ne}**.
* **The "High {theme_left} / High {theme_top}" Territory:** Currently dominated by **{winner_nw}**.
* **The "High {theme_right} / Low {theme_top}" Territory:** Currently dominated by **{winner_se}**.
* **The "High {theme_left} / Low {theme_top}" Territory:** Currently dominated by **{winner_sw}**.
"""

st.markdown(explanation)
