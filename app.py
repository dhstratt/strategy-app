import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import collections

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Universal Strategy Engine")

# --- MAIN APP ---
st.title("🧠 The Strategy Engine")
st.markdown("Upload **Core Data** first. Then overlay **Passive Brands** OR **Passive Attributes**.")

# --- SIDEBAR: ACTIVE DATA ---
st.sidebar.header("1. The Core Map (Active)")
uploaded_file = st.sidebar.file_uploader("Upload Core Data", type=["csv", "xlsx"], key="active")

# --- SIDEBAR: PASSIVE DATA ---
st.sidebar.markdown("---")
st.sidebar.header("2. Passive Layers")
st.sidebar.caption("Upload files with new Brands (Columns) OR new Statements (Rows). The app will auto-detect.")
passive_files = st.sidebar.file_uploader("Upload Passive Files", type=["csv", "xlsx"], accept_multiple_files=True, key="passive")

# --- HELPER: LOAD & CLEAN ---
def load_file(file):
    if file.name.endswith('.csv'): return pd.read_csv(file)
    else: return pd.read_excel(file)

def clean_df(df):
    label_col = df.columns[0]
    df = df.set_index(label_col)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    # Total Assassin
    df = df[~df.index.astype(str).str.contains("Total|Universe|Base|Sample", case=False, regex=True)]
    valid_cols = [c for c in df.columns if "total" not in str(c).lower() and "base" not in str(c).lower()]
    return df[valid_cols]

if uploaded_file is None:
    st.info("👈 Waiting for Core Data upload...")
    st.stop()

# 1. PROCESS CORE DATA
df_active_raw = load_file(uploaded_file)
st.sidebar.success(f"Core: {uploaded_file.name} loaded.")

label_col = df_active_raw.columns[0]
all_brands = df_active_raw.columns[1:].tolist()
data_cols = st.sidebar.multiselect("Select Core Brands", all_brands, default=all_brands)

if not data_cols:
    st.warning("Please select at least one core brand.")
    st.stop()

df_active = clean_df(df_active_raw[ [label_col] + data_cols ])
df_active = df_active.loc[(df_active != 0).any(axis=1)]
df_active = df_active.loc[:, (df_active != 0).any(axis=0)]

if df_active.empty:
    st.error("Active data is empty.")
    st.stop()

# 2. THE MATH (SVD on Active)
N = df_active.values
P = N / N.sum()
r = P.sum(axis=1)
c = P.sum(axis=0)
E = np.outer(r, c)
E[E < 1e-9] = 1e-9
R = (P - E) / np.sqrt(E)
U, s, Vh = np.linalg.svd(R, full_matrices=False)

# Active Coordinates
row_coords = (U * s) / np.sqrt(r[:, np.newaxis])
col_coords = (Vh.T * s) / np.sqrt(c[:, np.newaxis])

# Store Core Results
df_brands = pd.DataFrame(col_coords[:, :2], columns=['x', 'y'])
df_brands['Label'] = df_active.columns
df_brands['Type'] = 'Brand (Core)'
df_brands['Distinctiveness'] = np.sqrt(df_brands['x']**2 + df_brands['y']**2)

df_attrs = pd.DataFrame(row_coords[:, :2], columns=['x', 'y'])
df_attrs['Label'] = df_active.index
df_attrs['Type'] = 'Attribute'
df_attrs['Distinctiveness'] = np.sqrt(df_attrs['x']**2 + df_attrs['y']**2)

# 3. PROCESS PASSIVE LAYERS (Bi-Directional)
passive_layer_data = []

if passive_files:
    for p_file in passive_files:
        try:
            p_raw = load_file(p_file)
            p_clean = clean_df(p_raw)
            
            # --- DETECTIVE WORK: Is it Brands or Attributes? ---
            # Check overlap with Active Columns (Brands) vs Active Index (Attributes)
            
            # Intersection with Active Brands (Columns)
            common_brands = list(set(p_clean.columns) & set(df_active.columns))
            # Intersection with Active Attributes (Rows)
            common_attrs = list(set(p_clean.index) & set(df_active.index))
            
            # LOGIC A: PASSIVE ATTRIBUTES (Rows)
            # If the file shares many COLUMNS with Active, it's adding new rows.
            if len(common_brands) > len(common_attrs):
                layer_type = "Attribute"
                
                # Align columns to match Active Map's Brands
                p_aligned = p_clean[common_brands].reindex(columns=df_active.columns).fillna(0)
                
                # Formula: F_sup = (P_sup * G_active) / s
                # Need Row Profiles (Divide by row sum)
                row_sums = p_aligned.sum(axis=1)
                row_sums[row_sums == 0] = 1
                p_profiles = p_aligned.div(row_sums, axis=0)
                
                # Project onto Column Coordinates (Brands)
                # (NewAttrs x Brands) @ (Brands x Dims)
                proj = p_profiles.values @ col_coords[:, :2]
                proj = proj / s[:2]
                
                res = pd.DataFrame(proj, columns=['x', 'y'])
                res['Label'] = p_aligned.index
                res['Type'] = f"Layer (Attr): {p_file.name}"
                res['LayerName'] = p_file.name
                res['Shape'] = 'star' # Visual cue
                
            # LOGIC B: PASSIVE BRANDS (Columns)
            # If the file shares many ROWS with Active, it's adding new columns.
            else:
                layer_type = "Brand"
                
                # Align rows to match Active Map's Attributes
                p_aligned = p_clean.reindex(df_active.index).fillna(0)
                
                # Formula: G_sup = (P_sup.T * F_active) / s
                # Need Col Profiles
                col_sums = p_aligned.sum(axis=0)
                col_sums[col_sums == 0] = 1
                p_profiles = p_aligned.div(col_sums, axis=1)
                
                # Project onto Row Coordinates (Attributes)
                # (Brands x OldAttrs) @ (OldAttrs x Dims)
                proj = p_profiles.T.values @ row_coords[:, :2]
                proj = proj / s[:2]
                
                res = pd.DataFrame(proj, columns=['x', 'y'])
                res['Label'] = p_aligned.columns
                res['Type'] = f"Layer (Brand): {p_file.name}"
                res['LayerName'] = p_file.name
                res['Shape'] = 'diamond' # Visual cue

            res['Distinctiveness'] = np.sqrt(res['x']**2 + res['y']**2)
            passive_layer_data.append(res)

        except Exception as e:
            st.sidebar.error(f"Error projecting {p_file.name}: {e}")

# --- THEME SUGGESTION ---
def suggest_theme(df, direction, n=4):
    if direction == 'Right': subset = df.sort_values('x', ascending=False).head(n)['Label']
    elif direction == 'Left': subset = df.sort_values('x', ascending=True).head(n)['Label']
    elif direction == 'Top': subset = df.sort_values('y', ascending=False).head(n)['Label']
    elif direction == 'Bottom': subset = df.sort_values('y', ascending=True).head(n)['Label']
    top_1 = subset.iloc[0]
    all_text = " ".join(subset.astype(str)).lower()
    stop_words = ['i', 'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'my', 'for', 'on', 'with', 'often', 'prefer', 'like', 'typically', 'look', 'buy', 'make', 'feel', 'more', 'less', 'most', 'brand']
    words = [w.strip(".,()&") for w in all_text.split() if w.strip(".,()&") not in stop_words and len(w) > 2]
    if words:
        most_common = collections.Counter(words).most_common(1)[0]
        if most_common[1] > 1: return most_common[0].capitalize() 
    return (top_1[:20] + '..') if len(top_1) > 20 else top_1

sugg_right = suggest_theme(df_attrs, 'Right')
sugg_left = suggest_theme(df_attrs, 'Left')
sugg_top = suggest_theme(df_attrs, 'Top')
sugg_bottom = suggest_theme(df_attrs, 'Bottom')

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("---")
st.sidebar.subheader("🏷️ Axis Interpretation")
theme_left = st.sidebar.text_input("← Left Axis", value=sugg_left)
theme_right = st.sidebar.text_input("Right Axis →", value=sugg_right)
theme_bottom = st.sidebar.text_input("↓ Bottom Axis", value=sugg_bottom)
theme_top = st.sidebar.text_input("Top Axis ↑", value=sugg_top)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Layer Controls")
show_core = st.sidebar.checkbox("Show Core Brands", value=True)
show_attrs = st.sidebar.checkbox("Show Core Attributes", value=True)

active_layers = []
if passive_layer_data:
    st.sidebar.caption("Toggle Passive Layers:")
    for layer in passive_layer_data:
        name = layer['LayerName'].iloc[0]
        if st.sidebar.checkbox(f"Show {name}", value=True):
            active_layers.append(layer)

st.sidebar.markdown("---")
n_brands_show = st.sidebar.slider("Core Density", 2, len(df_brands), len(df_brands))
n_attrs_show = st.sidebar.slider("Attribute Density", 5, len(df_attrs), 15)

all_labels = df_brands['Label'].tolist()
for layer in active_layers: all_labels += layer['Label'].tolist()
focus_brand = st.sidebar.selectbox("Highlight/Story Mode:", ["None"] + all_labels)

# --- FILTERING ---
top_brands = df_brands.sort_values('Distinctiveness', ascending=False).head(n_brands_show)
top_attrs = df_attrs.sort_values('Distinctiveness', ascending=False).head(n_attrs_show)

# --- CHART ---
fig = go.Figure()

core_color = '#1f77b4'
attr_color = '#d62728'
passive_colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
dim_color = '#d3d3d3'

# 1. CORE BRANDS
if show_core:
    color_map = [core_color if (focus_brand == "None" or x == focus_brand) else dim_color for x in top_brands['Label']]
    opacity_map = [1.0 if (focus_brand == "None" or x == focus_brand) else 0.2 for x in top_brands['Label']]
    fig.add_trace(go.Scatter(
        x=top_brands['x'], y=top_brands['y'], mode='markers', name='Core Brands',
        marker=dict(size=14, color=color_map, opacity=opacity_map, line=dict(width=1, color='white')),
        text=top_brands['Label'], hovertemplate="<b>%{text}</b> (Core)<extra></extra>"
    ))

# 2. PASSIVE LAYERS
for i, layer in enumerate(active_layers):
    l_color = passive_colors[i % len(passive_colors)]
    if focus_brand != "None":
        p_color_map = [l_color if x == focus_brand else dim_color for x in layer['Label']]
        p_opacity = [1.0 if x == focus_brand else 0.2 for x in layer['Label']]
    else:
        p_color_map, p_opacity = l_color, 0.9

    # Symbol Logic: Diamond for Brands, Star for Attributes
    symbol = layer['Shape'].iloc[0] # diamond or star
    
    fig.add_trace(go.Scatter(
        x=layer['x'], y=layer['y'], mode='markers', name=layer['LayerName'].iloc[0],
        marker=dict(size=12, symbol=symbol, color=p_color_map, opacity=p_opacity, line=dict(width=1, color='white')),
        text=layer['Label'], hovertemplate="<b>%{text}</b> (Layer)<extra></extra>"
    ))

# 3. CORE ATTRIBUTES
if show_attrs:
    if focus_brand != "None":
        # Hero Coords Logic
        hero_row = None
        if focus_brand in df_brands['Label'].values: hero_row = df_brands[df_brands['Label'] == focus_brand].iloc[0]
        else:
            for l in active_layers:
                if focus_brand in l['Label'].values:
                    hero_row = l[l['Label'] == focus_brand].iloc[0]; break
        
        if hero_row is not None:
            top_attrs['Dist'] = np.sqrt((top_attrs['x'] - hero_row['x'])**2 + (top_attrs['y'] - hero_row['y'])**2)
            related = top_attrs.sort_values('Dist').head(5)['Label'].tolist()
            a_colors = [attr_color if x in related else dim_color for x in top_attrs['Label']]
            a_opacity = [1.0 if x in related else 0.2 for x in top_attrs['Label']]
        else: a_colors, a_opacity = attr_color, 0.7
    else: a_colors, a_opacity = attr_color, 0.7

    fig.add_trace(go.Scatter(
        x=top_attrs['x'], y=top_attrs['y'], mode='markers', name='Core Attributes',
        marker=dict(size=6, color=a_colors, opacity=a_opacity),
        text=top_attrs['Label'], hovertemplate="<b>%{text}</b><extra></extra>"
    ))

# 4. ANNOTATIONS
annotations = []
# Core
if show_core:
    for _, row in top_brands.iterrows():
        if focus_brand != "None" and row['Label'] != focus_brand: continue
        annotations.append(dict(x=row['x'], y=row['y'], text=row['Label'], showarrow=True, arrowhead=0, arrowcolor=core_color, ax=0, ay=-25, font=dict(size=13, color=core_color, family="Arial Black"), bgcolor="rgba(255,255,255,0.8)"))

# Passive
for i, layer in enumerate(active_layers):
    l_color = passive_colors[i % len(passive_colors)]
    for _, row in layer.iterrows():
        if focus_brand != "None" and row['Label'] != focus_brand: continue
        annotations.append(dict(x=row['x'], y=row['y'], text=row['Label'], showarrow=True, arrowhead=0, arrowcolor=l_color, ax=0, ay=-25, font=dict(size=12, color=l_color, family="Arial Black"), bgcolor="rgba(255,255,255,0.8)"))

# Attributes
if show_attrs:
    for _, row in top_attrs.iterrows():
        if focus_brand != "None" and a_opacity[top_attrs.index.get_loc(row.name)] < 0.5: continue
        annotations.append(dict(x=row['x'], y=row['y'], text=row['Label'], showarrow=True, arrowhead=0, arrowcolor=attr_color, ax=0, ay=-15, font=dict(size=10, color=attr_color), bgcolor="rgba(255,255,255,0.6)"))

fig.update_layout(
    annotations=annotations,
    title={'text': "Strategic Perceptual Map", 'y':0.95, 'x':0.5, 'xanchor': 'center'},
    template="plotly_white", height=800,
    xaxis=dict(title=f"← {theme_left} ... {theme_right} →", title_font=dict(size=16, family="Arial Black"), zeroline=True, showgrid=False),
    yaxis=dict(title=f"← {theme_bottom} ... {theme_top} →", title_font=dict(size=16, family="Arial Black"), zeroline=True, showgrid=False),
    showlegend=True, dragmode='pan'
)

fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=10, fillcolor="blue", opacity=0.03, layer="below", line_width=0)
fig.add_shape(type="rect", x0=-10, y0=-10, x1=0, y1=0, fillcolor="blue", opacity=0.03, layer="below", line_width=0)

st.plotly_chart(fig, use_container_width=True, config={'editable': True, 'scrollZoom': True})

st.subheader("📝 Strategic Interpretation")
st.info(f"Analysis based on {len(df_active)} Core Brands. Passive layers are projected onto this existing reality.")
st.markdown(f"**Map Logic:**\n* **Horizontal:** {theme_left} vs. {theme_right}\n* **Vertical:** {theme_top} vs. {theme_bottom}")
