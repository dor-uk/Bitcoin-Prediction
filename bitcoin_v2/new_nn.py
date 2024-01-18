import torch
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
from feature_eng_v2 import neural_engineering, feature_engineering
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import plotly.graph_objects as go

# Set the path to the main folder
general_path = os.path.abspath(os.path.dirname(__file__))

# Read the dataset
data = pd.read_csv(os.path.join(general_path, "Btc_small.csv"))
internet_data = pd.read_csv(os.path.join(general_path, "BTC-USD.csv"))
internet_data = internet_data[:len(data)]

# Perform feature engineering on the dataset
data, feature_columns, target_column = neural_engineering(data, internet_data)


# Split the data into train, validation, and test sets
train_size = len(data) - 120
val_size = len(data) - 60
train = data[:train_size + 1]
valid = data[train_size:val_size + 1]
test = data[val_size:]

# Extract the features and target variable for each set
X_train = train[feature_columns]
Y_train = train[target_column]

X_val = valid[feature_columns]
y_val = valid[target_column]

x_test = test[feature_columns]
y_test = test[target_column]

# Normalize the data using StandardScaler
mScalar = StandardScaler()
mScalar.fit(X_train)
X_train_scaled = mScalar.fit_transform(X_train)
X_val_scaled = mScalar.transform(X_val)
X_test_scaled = mScalar.transform(x_test)

# Convert the numpy arrays to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train_scaled).float()
X_val_tensor = torch.from_numpy(X_val_scaled).float()
X_test_tensor = torch.from_numpy(X_test_scaled).float()

y_train_tensor = torch.from_numpy(Y_train.values).float() / 1000
y_val_tensor = torch.from_numpy(y_val.values).float() / 1000
y_test_tensor = torch.from_numpy(y_test.values).float() / 1000

# Define the MLP model class
class mMLP(torch.nn.Module):
    def __init__(self, input_dim, out_dim):
        super(mMLP, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, 32)
        self.layer2 = torch.nn.Linear(32, 32)
        self.layer3 = torch.nn.Linear(32, 32)
        self.layer4 = torch.nn.Linear(32, out_dim)

        self.act_func = torch.nn.LeakyReLU()

    def forward(self, x_):
        out1 = self.layer1(x_)
        out1 = self.act_func(out1)

        out1 = self.layer2(out1)
        out1 = self.act_func(out1)

        out1 = self.layer3(out1)
        out1 = self.act_func(out1)

        out1 = self.layer4(out1)
        return out1

# Hyperparameters
input_size = 26  
hidden_size = 64
num_layers = 2
output_size = 1
batch_size = 32

# Seed for reproducibility
torch.manual_seed(42)

# Create the MLP model
mModel = mMLP(input_dim=X_train_tensor.shape[1], out_dim=1)

# Set the maximum number of epochs for training
MAX_EPOCH = 10000

# Create the optimizer and loss function
mOptimizer = torch.optim.Adam(params=mModel.parameters(), lr=5.7e-2)
mLoss = torch.nn.MSELoss()

# Lists to store training and validation losses during training
val_losses_list, train_losses_list = [], []

# Variables to keep track of the best model and the best validation loss
best_model = None
min_val_loss = np.inf

# Training loop
for epoch_indx in range(MAX_EPOCH):
    mModel.train()
    train_preds = mModel.forward(X_train_tensor)

    # Compute the training loss and store it
    loss = mLoss(train_preds, y_train_tensor)
    train_losses_list.append(loss.item())
    
    # Print the training loss every 100 epochs
    if epoch_indx % 100 == 0:
        print(f"Epoch {epoch_indx:4d}: Train Loss: {loss.item():.8f}", end=" ")

    # Perform backpropagation and update the model's parameters
    mOptimizer.zero_grad()
    loss.backward()
    mOptimizer.step()

    # Validation
    with torch.no_grad():
        mModel.eval()
        val_preds = mModel.forward(X_val_tensor)
        val_loss = mLoss(val_preds, y_val_tensor)
        val_losses_list.append(val_loss.item())

        # Print the validation loss and update the best model
        if epoch_indx % 100 == 0:
            min_val_loss_index = np.argmin(np.asarray(val_losses_list))
            min_val_loss = np.min(np.asarray(val_losses_list))
            print(f"Val Loss: {val_loss.item():.8f}, Min Val Loss is at Epoch: {min_val_loss_index:4d}, : {min_val_loss:.8f}")    
        
        if min_val_loss > val_loss.item():
            min_val_loss = val_loss.item()
            best_model = copy.deepcopy(mModel)

# Plot the training and validation losses
plt.plot(train_losses_list)
plt.plot(val_losses_list)
plt.show()

# Perform predictions using the best model on train, validation, and test sets
with torch.no_grad():
    train_preds = best_model.forward(X_train_tensor).numpy() * 1000
    train_preds = pd.DataFrame(train_preds, index=X_train.index)
    
    val_preds = best_model.forward(X_val_tensor).numpy() * 1000
    val_preds = pd.DataFrame(val_preds, index=X_val.index)
    
    test_preds = best_model.forward(X_test_tensor).numpy() * 1000
    test_preds = pd.DataFrame(test_preds, index=x_test.index)

# Calculate RMSE for train and test sets
train_rmse = np.sqrt(mean_squared_error(Y_train, train_preds))
valid_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

# Calculate R2 score for train and test sets
train_r2 = r2_score(Y_train, train_preds)
valid_r2 = r2_score(y_val, val_preds)
test_r2 = r2_score(y_test, test_preds)

# Create the plotly figure for visualization
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=train_preds.index, y=train_preds[0],  name="Train Predictions", line=dict(color='dodgerblue'))
)
fig.add_trace(
    go.Scatter(x=val_preds.index, y=val_preds[0],  name="Validation Predictions", line=dict(color='magenta'))
)
fig.add_trace(
    go.Scatter(x=test_preds.index, y=test_preds[0],  name="Test Predictions", line=dict(color='orange'))
)
fig.add_trace(
    go.Scatter(x=pd.DataFrame(Y_train).index, y=pd.DataFrame(Y_train)["Open"],  name="Train Target", line=dict(color='brown'))
)
fig.add_trace(
    go.Scatter(x=pd.DataFrame(y_val).index, y=pd.DataFrame(y_val)["Open"],  name="Validation Target", line=dict(color='black'))
)
fig.add_trace(
    go.Scatter(x=pd.DataFrame(y_test).index, y=pd.DataFrame(y_test)["Open"],  name="Test Target", line=dict(color='green'))
)
fig.update_layout(height=600)
fig.show()



print(f"Train RMSE: {train_rmse:.4f}")
print(f"Valid RMSE: {valid_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Train R2 Score: {train_r2:.4f}")
print(f"Valid R2 Score: {valid_r2:.4f}")
print(f"Test R2 Score: {test_r2:.4f}")
