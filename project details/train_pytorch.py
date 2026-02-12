# step7_train_pytorch.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

# Set working directory
working_dir = 'F:\project details'
print("Step 7 - Current Working Directory:", os.getcwd())
for file in ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']:
    file_path = os.path.join(working_dir, file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found! Run Step 6 first.")

# Load data
X_train = pd.read_csv(os.path.join(working_dir, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(working_dir, 'y_train.csv')).values
X_test = pd.read_csv(os.path.join(working_dir, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(working_dir, 'y_test.csv')).values

# Feature selection: Remove low-variance features
variance = X_train.var()
low_variance_cols = variance[variance < 1e-6].index
print(f"\nStep 7 - Removing low-variance columns: {list(low_variance_cols)}")
X_train = X_train.drop(columns=low_variance_cols)
X_test = X_test.drop(columns=low_variance_cols)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Check for NaN/Inf
print("\nStep 7 - NaN or Inf in input tensors:")
print("X_train NaN count:", torch.isnan(X_train_tensor).sum().item())
print("X_train Inf count:", torch.isinf(X_train_tensor).sum().item())
print("y_train NaN count:", torch.isnan(y_train_tensor).sum().item())
print("y_train Inf count:", torch.isinf(y_train_tensor).sum().item())
print("X_test NaN count:", torch.isnan(X_test_tensor).sum().item())
print("X_test Inf count:", torch.isinf(X_test_tensor).sum().item())
print("y_test NaN count:", torch.isnan(y_test_tensor).sum().item())
print("y_test Inf count:", torch.isinf(y_test_tensor).sum().item())

# Check y_test variance
y_test_var = torch.var(y_test_tensor)
print("\nStep 7 - y_test variance:", y_test_var.item())
if y_test_var < 1e-6:
    print("Step 7 - Warning: y_test has near-zero variance, R² may be undefined")

# Define neural network
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(64)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.xavier_uniform_(self.fc6.weight)

    def forward(self, x):
        x = self.batch_norm1(self.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.batch_norm2(self.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.batch_norm3(self.relu(self.fc3(x)))
        x = self.dropout(x)
        x = self.batch_norm4(self.relu(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        return x

# Initialize model
model = Net(X_train_tensor.shape[1])
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.000005, weight_decay=1e-4)

# Create DataLoader for batch training
dataset = TensorDataset(X_train_tensor, y_train_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Train model
model.train()
num_epochs = 3000
for epoch in range(num_epochs):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print(f"Step 7 - Warning: NaN or Inf in outputs at epoch {epoch}")
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
        loss = criterion(outputs, batch_y)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Step 7 - Warning: NaN or Inf loss at epoch {epoch}")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
    if epoch % 200 == 0:
        with torch.no_grad():
            full_outputs = model(X_train_tensor)
            full_loss = criterion(full_outputs, y_train_tensor)
        print(f"Epoch {epoch}, Loss: {full_loss.item()}")

# Evaluate
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
        print("Step 7 - Warning: NaN or Inf in predictions")
        y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    mse = criterion(y_pred, y_test_tensor).item()
    y_mean = y_test_tensor.mean()
    tss = ((y_test_tensor - y_mean) ** 2).sum().item()
    r2 = 1 - mse / (tss / y_test_tensor.shape[0]) if tss > 1e-6 else float('nan')
    print("\nStep 7 - Model MSE:", mse)
    print("Step 7 - Model R² Score:", r2)

# Save model
model_path = os.path.join(working_dir, 'cesium_model_pytorch.pth')
torch.save(model.state_dict(), model_path)
print(f"\nStep 7 - Model saved to {model_path}")