import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
import os
from datetime import datetime
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

working_dir = 'F:\\project detail'
os.chdir(working_dir)

# --- Load latest train/test ---
X_train_file = max(glob('X_train_*.csv'), key=os.path.getctime)
X_test_file = max(glob('X_test_*.csv'), key=os.path.getctime)
y_train_file = max(glob('y_train_*.csv'), key=os.path.getctime)
y_test_file = max(glob('y_test_*.csv'), key=os.path.getctime)

X_train = pd.read_csv(X_train_file).fillna(-999)
X_test = pd.read_csv(X_test_file).fillna(-999)
y_train = pd.read_csv(y_train_file)['Activity_concentration'].fillna(0).values
y_test = pd.read_csv(y_test_file)['Activity_concentration'].fillna(0).values

# --- Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for CNN: (samples, channels=1, features)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- Define CNN Model ---
class CNNRegressor(nn.Module):
    def __init__(self, input_size):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3)
        self.fc1 = nn.Linear((input_size - 4) * 32, 64)  # (input_size - 2* (kernel-1))
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

input_size = X_train_tensor.shape[2]
model = CNNRegressor(input_size).to(device)

# --- Loss & Optimizer ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Train ---
num_epochs = 50
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    batch_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_losses.append(np.mean(batch_losses))

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor.to(device))
        val_loss = criterion(val_outputs, y_test_tensor.to(device)).item()
        val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}")

# --- Predictions ---
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor.to(device)).cpu().numpy().flatten()

cnn_mse = mean_squared_error(y_test, y_pred)
cnn_r2 = r2_score(y_test, y_pred)
print(f"CNN (PyTorch) - MSE: {cnn_mse:.4f}, R²: {cnn_r2:.4f}")

# --- Save ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pred_df = pd.DataFrame({'Actual': y_test, 'CNN_Predicted': y_pred})
pred_df.to_csv(f'cnn_pytorch_predictions_{timestamp}.csv', index=False)

torch.save(model.state_dict(), f'cnn_pytorch_model_{timestamp}.pth')
joblib.dump(scaler, f'cnn_pytorch_scaler_{timestamp}.joblib')

# --- Plots ---
# Training history
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('CNN (PyTorch) Training History')
plt.legend()
plt.savefig(f'cnn_pytorch_training_loss_{timestamp}.png')
plt.close()

# Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('CNN (PyTorch) - Actual vs Predicted')
plt.savefig(f'cnn_pytorch_accuracy_{timestamp}.png')
plt.close()

# Metrics
plt.figure()
metrics = {'MSE': cnn_mse, 'R²': cnn_r2}
plt.bar(metrics.keys(), metrics.values(), color=['orange', 'green'])
plt.title('CNN (PyTorch) Model Metrics')
plt.ylabel('Value')
plt.savefig(f'cnn_pytorch_metrics_{timestamp}.png')
plt.close()
