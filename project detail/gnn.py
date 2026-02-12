import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
import os
from datetime import datetime
import joblib

# --- Working Directory ---
working_dir = 'F:\\project detail'
os.chdir(working_dir)

# --- Load latest train/test files ---
X_train_file = max(glob('X_train_*.csv'), key=os.path.getctime)
X_test_file = max(glob('X_test_*.csv'), key=os.path.getctime)
y_train_file = max(glob('y_train_*.csv'), key=os.path.getctime)
y_test_file = max(glob('y_test_*.csv'), key=os.path.getctime)

X_train = pd.read_csv(X_train_file).fillna(-999).values
X_test = pd.read_csv(X_test_file).fillna(-999).values
y_train = pd.read_csv(y_train_file)['Activity_concentration'].fillna(0).values
y_test = pd.read_csv(y_test_file)['Activity_concentration'].fillna(0).values

# --- Memory-efficient graph ---
def build_graph_efficient(X, y=None):
    x = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.float) if y is not None else None
    return Data(x=x, y=y_tensor)

train_data = build_graph_efficient(X_train, y_train)
test_data = build_graph_efficient(X_test, y_test)

# --- Batch data for memory efficiency ---
def batch_data(data, batch_size=1024):
    graphs = []
    n_nodes = data.x.size(0)
    for i in range(0, n_nodes, batch_size):
        x_batch = data.x[i:i+batch_size]
        y_batch = data.y[i:i+batch_size]
        graphs.append(Data(x=x_batch, y=y_batch))
    return graphs

train_batches = batch_data(train_data, batch_size=1024)
test_batches = batch_data(test_data, batch_size=1024)

# --- Deeper GCN Model ---
class DeepGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, dropout=0.3):
        super(DeepGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, data):
        n_nodes = data.x.size(0)
        row, col = torch.meshgrid(torch.arange(n_nodes), torch.arange(n_nodes), indexing='ij')
        edge_index = torch.stack([row.flatten(), col.flatten()], dim=0).to(data.x.device)

        x = data.x
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.squeeze()

# --- Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepGCN(X_train.shape[1], 128, 1, num_layers=3, dropout=0.3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

num_epochs = 100
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    for batch in train_batches:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.x.size(0)
    avg_loss = total_loss / X_train.shape[0]
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

# --- Evaluation ---
model.eval()
pred_list = []
y_true_list = []

with torch.no_grad():
    for batch in test_batches:
        batch = batch.to(device)
        pred = model(batch)
        pred_list.append(pred.cpu())
        y_true_list.append(batch.y.cpu())

pred = torch.cat(pred_list).numpy()
y_true = torch.cat(y_true_list).numpy()

mse = np.mean((pred - y_true)**2)
ss_total = np.sum((y_true - np.mean(y_true))**2)
ss_res = np.sum((y_true - pred)**2)
r2 = 1 - (ss_res / ss_total)

print(f"Deep GNN - MSE: {mse:.4f}, R²: {r2:.4f}")

# --- Save Predictions ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pred_df = pd.DataFrame({'Actual': y_true, 'GNN_Predicted': pred})
pred_df.to_csv(f'deep_gnn_predictions_{timestamp}.csv', index=False)

# --- Save Model ---
joblib.dump(model.state_dict(), f'deep_gnn_model_{timestamp}.pth')

# --- Plot Actual vs Predicted ---
plt.figure()
plt.scatter(y_true, pred, alpha=0.6, color='blue')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Deep GNN - Actual vs Predicted')
plt.savefig(f'deep_gnn_accuracy_{timestamp}.png')
plt.close()
