import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

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
    y_train_series = pd.read_csv('y_train.csv')['Activity_concentration']  # Series
    X_test = pd.read_csv('X_test.csv')
    y_test_series = pd.read_csv('y_test.csv')['Activity_concentration']  # Series
    print("Step 7 - Data loaded successfully.")
    print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except Exception as e:
    print(f"Unexpected error loading data: {e}")
    exit()

# Pre-fill NaN in raw data
y_train_series = y_train_series.fillna(y_train_series.mean())
y_test_series = y_test_series.fillna(y_test_series.mean())
y_train = y_train_series.values.reshape(-1, 1)  # Convert to NumPy after filling
y_test = y_test_series.values.reshape(-1, 1)    # Convert to NumPy after filling

# Feature selection: Remove low-variance features
variance = X_train.var()
low_variance_cols = variance[variance < 1e-6].index
print(f"\nStep 7 - Removing low-variance columns: {list(low_variance_cols)}")
X_train = X_train.drop(columns=low_variance_cols)
X_test = X_test.drop(columns=low_variance_cols)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Check for NaN/Inf (post-scaling)
print("\nStep 7 - NaN or Inf in input tensors after cleaning:")
print("X_train NaN count:", torch.isnan(X_train_tensor).sum().item())
print("X_train Inf count:", torch.isinf(X_train_tensor).sum().item())
print("y_train NaN count:", torch.isnan(y_train_tensor).sum().item())
print("y_train Inf count:", torch.isinf(y_train_tensor).sum().item())
print("X_test NaN count:", torch.isnan(X_test_tensor).sum().item())
print("X_test Inf count:", torch.isinf(X_test_tensor).sum().item())
print("y_test NaN count:", torch.isnan(y_test_tensor).sum().item())
print("y_test Inf count:", torch.isinf(y_test_tensor).sum().item())

# Check variance
y_train_var = torch.var(y_train_tensor)
y_test_var = torch.var(y_test_tensor)
print("\nStep 7 - y_train variance:", y_train_var.item())
print("Step 7 - y_test variance:", y_test_var.item())
if y_train_var < 1e-6 or y_test_var < 1e-6:
    print("Step 7 - Warning: Near-zero variance in y_train or y_test, R² may be unreliable")

# Split training data into train and validation sets
X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = train_test_split(
    X_train_tensor, y_train_tensor, test_size=0.2, random_state=42
)

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
        self.dropout = nn.Dropout(0.3)  # Reduced from 0.5
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(64)
        self.initialize_weights()

    def initialize_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

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
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Increased to 0.001
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Reduced to 64
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Early stopping and tracking
patience = 30
best_val_loss = float('inf')
epochs_no_improve = 0
train_mse_list = []
val_mse_list = []
test_mse_list = []
train_r2_list = []
val_r2_list = []
test_r2_list = []

# Train model
model.train()
num_epochs = 1000  # Reduced for efficiency
best_model_path = os.path.join(working_dir, f'best_cesium_model_pytorch_{timestamp}.pth')

for epoch in range(num_epochs):
    # Train
    model.train()
    train_batch_mse = []
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Step 7 - Warning: NaN or Inf loss at epoch {epoch}")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        train_batch_mse.append(loss.item())
    
    train_mse = np.mean(train_batch_mse)
    
    # Validate
    model.eval()
    val_batch_mse = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            val_batch_mse.append(criterion(outputs, batch_y).item())
        val_loss = np.mean(val_batch_mse)
    
    # Compute metrics
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        train_mse_full = criterion(train_outputs, y_train_tensor).item()
        train_y_mean = y_train_tensor.mean()
        train_tss = ((y_train_tensor - train_y_mean) ** 2).sum().item()
        train_r2 = max(0, 1 - train_mse_full / (train_tss / y_train_tensor.shape[0])) if train_tss > 1e-6 else 0.0
        
        val_outputs = model(X_val_tensor)
        val_mse_full = criterion(val_outputs, y_val_tensor).item()
        val_y_mean = y_val_tensor.mean()
        val_tss = ((y_val_tensor - val_y_mean) ** 2).sum().item()
        val_r2 = max(0, 1 - val_mse_full / (val_tss / y_val_tensor.shape[0])) if val_tss > 1e-6 else 0.0
        
        test_outputs = model(X_test_tensor)
        test_mse_full = criterion(test_outputs, y_test_tensor).item()
        test_y_mean = y_test_tensor.mean()
        test_tss = ((y_test_tensor - test_y_mean) ** 2).sum().item()
        test_r2 = max(0, 1 - test_mse_full / (test_tss / y_test_tensor.shape[0])) if test_tss > 1e-6 else 0.0
    
    train_mse_list.append(train_mse)
    val_mse_list.append(val_loss)
    test_mse_list.append(test_mse_full)
    train_r2_list.append(train_r2)
    val_r2_list.append(val_r2)
    test_r2_list.append(test_r2)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Train MSE: {train_mse:.4f}, Val MSE: {val_loss:.4f}, Test MSE: {test_mse_full:.4f}, "
              f"Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}")
    
    # Early stopping and model checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Step 7 - Saved best model to {best_model_path} at epoch {epoch}")
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
    
    # Update learning rate
    scheduler.step(val_loss)

# Load best model
model.load_state_dict(torch.load(best_model_path))

# Final evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    mse = criterion(y_pred, y_test_tensor).item()
    y_mean = y_test_tensor.mean()
    tss = ((y_test_tensor - y_mean) ** 2).sum().item()
    r2 = max(0, 1 - mse / (tss / y_test_tensor.shape[0])) if tss > 1e-6 else 0.0
    print("\nStep 7 - Model MSE:", mse)
    print("Step 7 - Model R² Score:", r2)

# Generate risk map
plt.figure(figsize=(10, 8))
sc = plt.scatter(
    X_test['Longitude_(deg)'], 
    X_test['Latitude_(deg)'], 
    c=y_pred.numpy().flatten(), 
    cmap='viridis', 
    s=50, 
    alpha=0.7
)
plt.colorbar(sc, label='Predicted Activity Concentration')
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('Risk Map: Predicted Activity Concentration')
plt.tight_layout()
plt.savefig(f'risk_map_{timestamp}_pytorch.png')
plt.close()
print(f"Step 7 - Risk map saved to risk_map_{timestamp}_pytorch.png")

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': y_test.flatten(),
    'PyTorch_Predicted': y_pred.numpy().flatten()
})
predictions_df.to_csv(f'pytorch_predictions_{timestamp}.csv', index=False)
print(f"\nStep 7 - Predictions saved to pytorch_predictions_{timestamp}.csv")

# Save model
model_path = os.path.join(working_dir, f'cesium_model_pytorch_{timestamp}.pth')
torch.save(model.state_dict(), model_path)
print(f"Step 7 - Model saved to {model_path}")

# Compute feature importances (gradient-based sensitivity)
model.eval()
X_test_tensor.requires_grad = True
outputs = model(X_test_tensor)
outputs.sum().backward()
gradients = X_test_tensor.grad.abs().mean(dim=0).numpy()
pytorch_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': gradients
}).sort_values(by='Importance', ascending=False)
pytorch_importance.to_csv(f'pytorch_feature_importance_{timestamp}.csv', index=False)
print(f"Step 7 - Feature importances saved to pytorch_feature_importance_{timestamp}.csv")

# Plot train vs test accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_mse_list, label='Train MSE', color='blue')
plt.plot(val_mse_list, label='Val MSE', color='green')
plt.plot(test_mse_list, label='Test MSE', color='orange')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Train vs Val vs Test MSE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_r2_list, label='Train R²', color='blue')
plt.plot(val_r2_list, label='Val R²', color='green')
plt.plot(test_r2_list, label='Test R²', color='orange')
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.title('Train vs Val vs Test R²')
plt.legend()

plt.tight_layout()
plt.savefig(f'accuracy_plot_{timestamp}_pytorch.png')
plt.close()
print(f"Step 7 - Accuracy plot saved to accuracy_plot_{timestamp}_pytorch.png")