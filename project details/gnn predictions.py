import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
 
import os
from datetime import datetime

# Set working directory
working_dir = 'F:\\project details'
os.chdir(working_dir)
print("Step 7 - Current Working Directory:", os.getcwd())

# Timestamp for unique file names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load data
try:
    for file in ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']:
        file_path = os.path.join(working_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found! Run Step 6 first.")
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv').values
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv').values
    print("Step 7 - Data loaded successfully.")
    print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Feature selection: Remove low-variance features
variance = X_train.var()
low_variance_cols = variance[variance < 1e-6].index
print(f"\nStep 7 - Removing low-variance columns: {list(low_variance_cols)}")
X_train = X_train.drop(columns=low_variance_cols)
X_test = X_test.drop(columns=low_variance_cols)

# Verify required columns
required_cols = ['Latitude_(deg)', 'Longitude_(deg)']
if not all(col in X_train.columns for col in required_cols):
    print(f"Error: Missing columns {required_cols}. Cannot proceed.")
    exit()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)
coords_train = scaler.fit_transform(X_train[['Latitude_(deg)', 'Longitude_(deg)']].values)
coords_test = scaler.transform(X_test[['Latitude_(deg)', 'Longitude_(deg)']].values)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Replace NaN/Inf
X_train_tensor = torch.nan_to_num(X_train_tensor, nan=0.0, posinf=0.0, neginf=0.0)
X_test_tensor = torch.nan_to_num(X_test_tensor, nan=0.0, posinf=0.0, neginf=0.0)
y_train_tensor = torch.nan_to_num(y_train_tensor, nan=0.0, posinf=0.0, neginf=0.0)
y_test_tensor = torch.nan_to_num(y_test_tensor, nan=0.0, posinf=0.0, neginf=0.0)

# Construct k-NN graph
k = 5  # First run's setting
edge_index_train = kneighbors_graph(coords_train, k, mode='connectivity', include_self=False).tocoo()
edge_index_test = kneighbors_graph(coords_test, k, mode='connectivity', include_self=False).tocoo()
edge_index_train = torch.tensor(np.vstack((edge_index_train.row, edge_index_train.col)), dtype=torch.long)
edge_index_test = torch.tensor(np.vstack((edge_index_test.row, edge_index_test.col)), dtype=torch.long)
edge_index_train = to_undirected(edge_index_train)
edge_index_test = to_undirected(edge_index_test)

# Create PyTorch Geometric Data objects
train_data = Data(x=X_train_tensor, edge_index=edge_index_train, y=y_train_tensor)
test_data = Data(x=X_test_tensor, edge_index=edge_index_test, y=y_test_tensor)

# Define GCN model (2 layers, as in first run)
class GCN(nn.Module):
    def __init__(self, input_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_size, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x

# Initialize model
model = GCN(X_train_tensor.shape[1])
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Early stopping and tracking
patience = 20
best_loss = float('inf')
epochs_no_improve = 0
train_mse_list = []
test_mse_list = []
train_r2_list = []
test_r2_list = []

# Train model
model.train()
num_epochs = 200
train_loader = DataLoader([train_data], batch_size=1, shuffle=False)
test_loader = DataLoader([test_data], batch_size=1, shuffle=False)

for epoch in range(num_epochs):
    # Train
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch)
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
        loss = criterion(outputs, batch.y)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Step 7 - Warning: NaN or Inf loss at epoch {epoch}")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
    
    # Compute train metrics
    model.eval()
    with torch.no_grad():
        train_outputs = model(train_data)
        train_outputs = torch.nan_to_num(train_outputs, nan=0.0, posinf=0.0, neginf=0.0)
        train_mse = criterion(train_outputs, y_train_tensor).item()
        train_y_mean = y_train_tensor.mean()
        train_tss = ((y_train_tensor - train_y_mean) ** 2).sum().item()
        train_r2 = 1 - train_mse / (train_tss / y_train_tensor.shape[0]) if train_tss > 1e-6 else float('nan')
        
        test_outputs = model(test_data)
        test_outputs = torch.nan_to_num(test_outputs, nan=0.0, posinf=0.0, neginf=0.0)
        test_mse = criterion(test_outputs, y_test_tensor).item()
        test_y_mean = y_test_tensor.mean()
        test_tss = ((y_test_tensor - test_y_mean) ** 2).sum().item()
        test_r2 = 1 - test_mse / (test_tss / y_test_tensor.shape[0]) if test_tss > 1e-6 else float('nan')
    
    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)
    train_r2_list.append(train_r2)
    test_r2_list.append(test_r2)
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}, Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    # Early stopping
    if loss.item() < best_loss:
        best_loss = loss.item()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Final evaluation
model.eval()
with torch.no_grad():
    y_pred = model(test_data)
    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    mse = criterion(y_pred, y_test_tensor).item()
    y_mean = y_test_tensor.mean()
    tss = ((y_test_tensor - y_mean) ** 2).sum().item()
    r2 = 1 - mse / (tss / y_test_tensor.shape[0]) if tss > 1e-6 else float('nan')
    print("\nStep 7 - GCN Model MSE:", mse)
    print("Step 7 - GCN Model R² Score:", r2)

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': y_test.flatten(),
    'GCN_Predicted': y_pred.numpy().flatten()
})
predictions_df.to_csv(f'gcn_predictions_{timestamp}.csv', index=False)
print(f"\nStep 7 - Predictions saved to gcn_predictions_{timestamp}.csv")

# Save model
model_path = os.path.join(working_dir, f'cesium_model_gcn_{timestamp}.pth')
torch.save(model.state_dict(), model_path)
print(f"Step 7 - Model saved to {model_path}")

# Compute feature importances
model.eval()
test_data.x.requires_grad = True
outputs = model(test_data)
outputs.sum().backward()
gradients = test_data.x.grad.abs().mean(dim=0).numpy()
gcn_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': gradients
}).sort_values(by='Importance', ascending=False)
gcn_importance.to_csv(f'gcn_feature_importance_{timestamp}.csv', index=False)
print(f"Step 7 - Feature importances saved to gcn_feature_importance_{timestamp}.csv")

#Plot train vs test accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_mse_list, label='Train MSE', color='blue')
plt.plot(test_mse_list, label='Test MSE', color='orange')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Train vs Test MSE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_r2_list, label='Train R²', color='blue')
plt.plot(test_r2_list, label='Test R²', color='orange')
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.title('Train vs Test R²')
plt.legend()

plt.tight_layout()
plt.savefig(f'accuracy_plot_{timestamp}_gnn.png')
plt.close()
print(f"Step 7 - Accuracy plot saved to accuracy_plot_{timestamp}.png")