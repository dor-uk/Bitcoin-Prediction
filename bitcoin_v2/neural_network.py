from feature_eng_v2 import feature_engineering 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


general_path = os.path.abspath(os.path.dirname(__file__))    #The path to the main folder



# Load and preprocess the Bitcoin data
def load_data():

    # read the dataset
    df = pd.read_csv(os.path.join(general_path,"Btc_small.csv"))
    internet_data = pd.read_csv(os.path.join(general_path,"BTC-USD.csv"))
    df = internet_data[:len(df)].copy()
    
    df.set_index('Date', inplace=True)
    # Choose the 'Open' price column as the target variable to predict
    target_col = 'Open'

    # Scale the data to the range [0, 1]
    scaler = MinMaxScaler()
    df[target_col] = scaler.fit_transform(df[[target_col]])

    # Extract features and target
    features = df.drop(columns=[target_col])
    target = df[target_col]
    
    # print(features.dtypes)
    # print(target.dtypes)

    # Convert data to PyTorch tensors
    features_tensor = torch.tensor(features.values, dtype=torch.float32)
    target_tensor = torch.tensor(target.values.reshape(-1, 1), dtype=torch.float32)

    return df, features_tensor, target_tensor


# Define the LSTM neural network architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Add one more dimension for sequence length
        x = x.unsqueeze(0)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Training function
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        num_batches = len(train_loader)
        train_mse = total_loss / num_batches
        rmse = train_mse**(1/2)
        #print(f'Train RMSE: {rmse}')

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}, Train RMSE: {rmse}")



# Main function to run the training and evaluation
def main():
    # Hyperparameters
    input_size = 5  
    hidden_size = 64
    num_layers = 2
    output_size = 1
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50
    
    # Load and preprocess the data
    data, features, target = load_data()
    # print(features)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Fit the scaler on the training data and transform both training and test data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.transform(y_test)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader for training data
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create the LSTM model
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs)
    
    # Evaluate the model (use the test data for evaluation)
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = criterion(test_predictions, y_test)
    print(f"Test Loss: {test_loss.item()}")
    #print(type(test_predictions))
    
    
    test_predictions = test_predictions.flatten()

    predictions = test_predictions.tolist()
    print(predictions)
    

    # dummies
    # dummies[:, 0] = test_predictions
    # dummies = scaler.inverse_transform(dummies)

    # test_predictions = (dummies[:, 0])
    #print(test_predictions)
    #print(test_predictions)


main()


