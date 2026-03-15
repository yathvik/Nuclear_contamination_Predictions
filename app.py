import streamlit as st
import os
import sys

st.title("Debug Mode")

current_dir = os.path.dirname(os.path.abspath(__file__))
st.write("Current directory:", current_dir)
st.write("Files here:", os.listdir(current_dir))

# Check for gnn_updated.py
gnn_path = os.path.join(current_dir, "gnn_updated.py")
st.write("gnn_updated.py exists:", os.path.exists(gnn_path))

# Try import
try:
    sys.path.insert(0, current_dir)
    from gnn_updated import ImprovedGCN, build_topk_edges
    st.success("✅ Import successful!")
except Exception as e:
    st.error(f"❌ Import failed: {str(e)}")
    st.code(str(e))