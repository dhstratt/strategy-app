import streamlit as st
import pandas as pd

# 1. The Title
st.title("My First Strategy Engine")
st.write("If you can see this text, you just built a web app!")

# 2. A Simple Interaction
st.header("The Simulator")
impact = st.slider("Select Brand Impact", 0, 100, 50)

if impact > 80:
    st.success("High Impact! Strategy is working.")
elif impact < 20:
    st.error("Warning: Strategy too weak.")
else:
    st.info("Strategy is stable.")

# 3. A Placeholder for your future map
st.subheader("Data Upload")
st.file_uploader("Drop your Cereal Data CSV here")
