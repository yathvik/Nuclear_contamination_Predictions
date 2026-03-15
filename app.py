import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from gnn_updated import ImprovedGCN, build_topk_edges

st.set_page_config(page_title="Cesium Contamination Prediction", layout="wide")

st.title("🌍 Cesium Contamination Prediction – Fukushima Dataset")
st.markdown("""
This app demonstrates **Graph Convolutional Network (GCN)** predictions for Cesium-137 contamination levels.
""")

DEFAULT_MODEL_PATH = "cesium_model_gcn_20260314_195717.pth"

with st.sidebar:
    st.header("Control Panel")
    use_uploaded = st.checkbox("Use uploaded files", value=False)
    X_train_file = st.file_uploader("X_train.csv", type="csv") if use_uploaded else None
    model_file = st.file_uploader("Model (.pth)", type=["pth", "pt"]) if use_uploaded else None
    show_test = st.checkbox("Show test predictions", value=False)
    k_neighbors = st.slider("Neighbors (k)", 5, 30, 10, step=5)
    run_button = st.button("Run Analysis", type="primary")

if run_button or "last_run" not in st.session_state:
    try:
        with st.spinner("Loading data..."):
            @st.cache_data
            def load_data():
                base = os.path.dirname(os.path.abspath(__file__))
                xt = pd.read_csv(os.path.join(base, "X_train.csv"))
                yt = pd.read_csv(os.path.join(base, "y_train.csv")).values.reshape(-1, 1)
                xe = pd.read_csv(os.path.join(base, "X_test.csv"))
                ye = pd.read_csv(os.path.join(base, "y_test.csv")).values.reshape(-1, 1)
                return xt, yt, xe, ye

            if use_uploaded and X_train_file:
                X_train = pd.read_csv(X_train_file)
                _, _, X_test, y_test = load_data()
                y_train = X_train.iloc[:, -1].values.reshape(-1, 1)
            else:
                X_train, y_train, X_test, y_test = load_data()

            st.success(f"✅ Loaded: {X_train.shape[0]:,} training samples, {X_train.shape[1]} features")

        with st.spinner("Building graph..."):
            feature_scaler = StandardScaler().fit(X_train.values)
            coord_scaler = StandardScaler().fit(X_train[['Latitude_(deg)', 'Longitude_(deg)']].values)
            y_scaler = StandardScaler().fit(y_train)

            X_scaled = feature_scaler.transform(X_train.values)
            coords = coord_scaler.transform(X_train[['Latitude_(deg)', 'Longitude_(deg)']].values)
            
            edge_index = build_topk_edges(X_scaled, coords, k=k_neighbors)
            train_data = Data(
                x=torch.tensor(X_scaled, dtype=torch.float32),
                edge_index=edge_index,
                y=torch.tensor(y_train, dtype=torch.float32)
            )

        with st.spinner("Loading model..."):
            model_path = model_file if model_file else os.path.join(current_dir, DEFAULT_MODEL_PATH)
            
            if not os.path.exists(model_path):
                st.error(f"Model not found: {model_path}")
                st.stop()

            model = ImprovedGCN(in_channels=X_scaled.shape[1])
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            st.info(f"Model: {os.path.basename(model_path)} | {params:,} parameters")

        with st.spinner("Predicting..."):
            with torch.no_grad():
                pred_scaled = model(train_data).squeeze().cpu().numpy()
                pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        st.subheader("Training Predictions (first 15)")
        df_results = pd.DataFrame({
            "Actual (Bq/kg)": y_train.flatten()[:15],
            "Predicted (Bq/kg)": pred[:15]
        }).round(2)
        st.dataframe(df_results, use_container_width=True)

        st.subheader("Predicted vs Actual")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_train.flatten(), pred, alpha=0.6, edgecolors='k', linewidths=0.5)
        ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='Ideal')
        ax.set_xlabel("Actual (Bq/kg)")
        ax.set_ylabel("Predicted (Bq/kg)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        if show_test:
            with st.spinner("Testing..."):
                X_test_s = feature_scaler.transform(X_test.values)
                coords_t = coord_scaler.transform(X_test[['Latitude_(deg)', 'Longitude_(deg)']].values)
                edge_t = build_topk_edges(X_test_s, coords_t, k=k_neighbors)
                test_data = Data(
                    x=torch.tensor(X_test_s, dtype=torch.float32),
                    edge_index=edge_t,
                    y=torch.tensor(y_test, dtype=torch.float32)
                )
                with torch.no_grad():
                    pred_t = y_scaler.inverse_transform(model(test_data).squeeze().cpu().numpy().reshape(-1, 1)).flatten()
                
                st.subheader("Test Predictions (first 15)")
                st.dataframe(pd.DataFrame({
                    "Actual": y_test.flatten()[:15],
                    "Predicted": pred_t[:15]
                }).round(2), use_container_width=True)

        st.success("✅ Complete!")
        st.session_state.last_run = True

    except Exception as e:
        st.error("Error:")
        st.exception(e)
else:
    st.info("Click **Run Analysis** to start")
