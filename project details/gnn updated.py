import os
from datetime import datetime

# --- Matplotlib headless backend (must be set before importing pyplot) ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Settings
# -------------------------
working_dir = r'F:\project details'   # change if needed
os.chdir(working_dir)
print("Working dir:", os.getcwd())
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Files expected
required_files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
for f in required_files:
    if not os.path.exists(os.path.join(working_dir, f)):
        raise FileNotFoundError(f"Missing file: {f}. Run previous steps or place file in {working_dir}")

# -------------------------
# Load data
# -------------------------
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.reshape(-1, 1)
y_test = pd.read_csv('y_test.csv').values.reshape(-1, 1)

print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# -------------------------
# Remove very low variance columns
# -------------------------
variance = X_train.var()
low_variance_cols = variance[variance < 1e-6].index.tolist()
print("Removing low-variance columns:", low_variance_cols)
X_train = X_train.drop(columns=low_variance_cols, errors='ignore')
X_test = X_test.drop(columns=low_variance_cols, errors='ignore')

# -------------------------
# Required coordinate columns check
# -------------------------
required_coords = ['Latitude_(deg)', 'Longitude_(deg)']
if not all(c in X_train.columns for c in required_coords):
    raise ValueError(f"Missing coordinate columns. Required: {required_coords}")

# -------------------------
# Scaling features and targets
# -------------------------
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train.values)
X_test_scaled = feature_scaler.transform(X_test.values)

coord_scaler = StandardScaler()
coords_train = coord_scaler.fit_transform(X_train[required_coords].values)
coords_test = coord_scaler.transform(X_test[required_coords].values)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# Convert to tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

# Remove NaNs/Infs
X_train_tensor = torch.nan_to_num(X_train_tensor, nan=0.0, posinf=0.0, neginf=0.0)
X_test_tensor = torch.nan_to_num(X_test_tensor, nan=0.0, posinf=0.0, neginf=0.0)
y_train_tensor = torch.nan_to_num(y_train_tensor, nan=0.0, posinf=0.0, neginf=0.0)
y_test_tensor = torch.nan_to_num(y_test_tensor, nan=0.0, posinf=0.0, neginf=0.0)

# -------------------------
# Build combined similarity graph
# -------------------------
def build_topk_edges(feature_matrix, coord_matrix, k=10, feat_weight=0.7, loc_weight=0.3):
    feat_sim = cosine_similarity(feature_matrix)
    loc_sim = cosine_similarity(coord_matrix)
    combined = feat_weight * feat_sim + loc_weight * loc_sim
    n = combined.shape[0]
    rows = []
    cols = []
    for i in range(n):
        idx = np.argsort(-combined[i])
        cnt = 0
        for j in idx:
            if j == i:
                continue
            rows.append(i)
            cols.append(j)
            cnt += 1
            if cnt >= k:
                break
    edge_index = np.vstack([rows, cols])
    return torch.tensor(edge_index, dtype=torch.long)

k_neighbors = 10
edge_index_train = build_topk_edges(X_train_scaled, coords_train, k=k_neighbors, feat_weight=0.75, loc_weight=0.25)
edge_index_test = build_topk_edges(X_test_scaled, coords_test, k=k_neighbors, feat_weight=0.75, loc_weight=0.25)

edge_index_train = to_undirected(edge_index_train).to(device)
edge_index_test = to_undirected(edge_index_test).to(device)

# -------------------------
# PyG Data objects
# -------------------------
train_data = Data(x=X_train_tensor, edge_index=edge_index_train, y=y_train_tensor)
test_data = Data(x=X_test_tensor, edge_index=edge_index_test, y=y_test_tensor)

# -------------------------
# Define improved GCN
# -------------------------
class ImprovedGCN(nn.Module):
    def __init__(self, in_channels, hidden1=128, hidden2=64, hidden3=32, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.conv3 = GCNConv(hidden2, hidden3)
        self.bn3 = nn.BatchNorm1d(hidden3)
        self.fc = nn.Linear(hidden3, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index)
        x1 = self.relu(self.bn1(x1))
        x1 = self.dropout(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = self.relu(self.bn2(x2))
        x2 = self.dropout(x2)
        x3 = self.conv3(x2, edge_index)
        x3 = self.relu(self.bn3(x3))
        out = self.fc(x3)
        return out

# Initialize model
model = ImprovedGCN(in_channels=X_train_tensor.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)  # Removed verbose

# -------------------------
# Training loop
# -------------------------
num_epochs = 400
patience = 40
best_val = float('inf')
epochs_no_improve = 0

train_mse_list = []
test_mse_list = []
train_r2_list = []
test_r2_list = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(train_data)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    loss = criterion(out, train_data.y)
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"NaN/Inf loss at epoch {epoch}, skipping")
        continue
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        train_pred = model(train_data).cpu().numpy().flatten()
        test_pred = model(test_data).cpu().numpy().flatten()
        train_pred_orig = y_scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
        test_pred_orig = y_scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
        y_train_orig = y_scaler.inverse_transform(y_train_scaled).flatten()
        y_test_orig = y_scaler.inverse_transform(y_test_scaled).flatten()

        train_mse = mean_squared_error(y_train_orig, train_pred_orig)
        test_mse = mean_squared_error(y_test_orig, test_pred_orig)
        train_r2 = max(0, r2_score(y_train_orig, train_pred_orig))  # Cap at 0
        test_r2 = max(0, r2_score(y_test_orig, test_pred_orig))    # Cap at 0

    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)
    train_r2_list.append(train_r2)
    test_r2_list.append(test_r2)

    scheduler.step(test_mse)

    if epoch % 20 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch:03d} | loss {loss.item():.4f} | train MSE {train_mse:.4f} R2 {train_r2:.4f} | test MSE {test_mse:.4f} R2 {test_r2:.4f}")

    # Early stopping
    if test_mse < best_val - 1e-8:
        best_val = test_mse
        epochs_no_improve = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= patience:
        print("Early stopping at epoch", epoch)
        break

# Restore best model
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

# Final evaluation
model.eval()
with torch.no_grad():
    y_pred_scaled = model(test_data).cpu().numpy().flatten()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = y_test.flatten()
    final_mse = mean_squared_error(y_true, y_pred)
    final_r2 = max(0, r2_score(y_true, y_pred))  # Cap at 0
    print("\nFinal GCN MSE:", final_mse, "R2:", final_r2)

# Save predictions
pred_df = pd.DataFrame({'Actual': y_true.flatten(), 'GCN_Predicted': y_pred.flatten()})
pred_csv = f'gcn_predictions_{timestamp}.csv'
pred_df.to_csv(pred_csv, index=False)
print("Predictions saved:", pred_csv)

# Save model
model_path = os.path.join(working_dir, f'cesium_model_gcn_{timestamp}.pth')
torch.save(model.state_dict(), model_path)
print("Model saved:", model_path)

# Feature importance via input gradients
train_data.x.requires_grad = True
model.zero_grad()
out = model(train_data)
loss_for_grad = out.sum()
loss_for_grad.backward()
grads = train_data.x.grad.abs().mean(dim=0).cpu().numpy()
feat_imp_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': grads})
feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)
feat_imp_csv = f'gcn_feature_importance_{timestamp}.csv'
feat_imp_df.to_csv(feat_imp_csv, index=False)
print("Feature importances saved:", feat_imp_csv)

# Plot training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_mse_list, label='Train MSE')
plt.plot(test_mse_list, label='Test MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.title('MSE')

plt.subplot(1, 2, 2)
plt.plot(train_r2_list, label='Train R2')
plt.plot(test_r2_list, label='Test R2')
plt.xlabel('Epochs')
plt.ylabel('R2')
plt.legend()
plt.ylim(0, 1)  # Restrict R² to [0, 1]
plt.title('R2')

plt.tight_layout()
plot_file = f'accuracy_plot_{timestamp}.png'
plt.savefig(plot_file)
plt.close()
print("Accuracy plot saved:", plot_file)

# Baseline: Random Forest
print("\nTraining RandomForest baseline for sanity check...")
rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train.ravel())
rf_pred = rf.predict(X_test_scaled)
rf_mse = mean_squared_error(y_test.ravel(), rf_pred)
rf_r2 = max(0, r2_score(y_test.ravel(), rf_pred))  # Cap at 0
print("RandomForest MSE:", rf_mse, "R2:", rf_r2)

rf_df = pd.DataFrame({'Actual': y_test.ravel(), 'RF_Predicted': rf_pred})
rf_csv = f'rf_predictions_{timestamp}.csv'
rf_df.to_csv(rf_csv, index=False)
print("RF predictions saved:", rf_csv)

print("\nDone.")