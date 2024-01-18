import torch
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import pandas as pd
from feature_eng_v2 import feature_engineering
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
import plotly.graph_objects as go


general_path = os.path.abspath(os.path.dirname(__file__))    #The path to the main folder

# read the dataset
data = pd.read_csv(os.path.join(general_path,"Btc_small.csv"))
internet_data = pd.read_csv(os.path.join(general_path,"BTC-USD.csv"))
internet_data = internet_data[:len(data)]


data, feature_columns, target_column = feature_engineering(data, internet_data)


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



# Normalize the data using MinMaxScaler
mScalar = MinMaxScaler()
mScalar.fit(X_train)
X_train_scaled = mScalar.transform(X_train)
X_val_scaled = mScalar.transform(X_val)
X_test_scaled = mScalar.transform(x_test)


y_scalar = MinMaxScaler()
y_scalar.fit(Y_train.values.reshape(-1, 1))
Y_train_scaled = y_scalar.transform(Y_train.values.reshape(-1, 1))
Y_valid_scaled = y_scalar.transform(y_val.values.reshape(-1, 1))
Y_test_scaled = y_scalar.transform(y_test.values.reshape(-1, 1))

# Here we'll create sequences like in the provided example
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 5

X_train, y_train = create_sequences(X_train_scaled, seq_length)
X_val, y_val = create_sequences(X_val_scaled, seq_length)
X_test, y_test = create_sequences(X_test_scaled, seq_length)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()



# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(LSTMModel, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden, num_layers=n_layers, dropout=0.5)
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
        )

    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(sequences.view(len(sequences), self.seq_len, -1), self.hidden)
        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred

# Define hyperparameters
input_size = len(feature_columns)  # Number of input features
hidden_size = 512
num_layers = 2
output_size = 1
learning_rate = 5e-4
#num_epochs = 100


# Seed for reproducibility
torch.manual_seed(42)

# Model hyperparameters
model = LSTMModel(n_features=input_size, n_hidden=hidden_size, seq_len=seq_length, n_layers=num_layers)
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 60
train_hist = np.zeros(num_epochs)
val_hist = np.zeros(num_epochs)

for t in range(num_epochs):
    model.reset_hidden_state()
    y_train_pred = model(X_train_tensor)
    loss = loss_fn(y_train_pred.float(), y_train_tensor)
    
    with torch.no_grad():
        y_val_pred = model(X_val_tensor)
        val_loss = loss_fn(y_val_pred.float(), y_val_tensor)
    val_hist[t] = val_loss.item()

    train_hist[t] = loss.item()

    if t % 10 == 0:
        print(f'Epoch {t} train loss: {loss.item()} validation loss: {val_loss.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plotting
plt.plot(train_hist, label="Training loss")
plt.plot(val_hist, label="Validation loss")
plt.legend()
plt.show()

