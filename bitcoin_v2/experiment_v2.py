import pandas as pd
from feature_eng_v2 import feature_engineering 
import os
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

general_path = os.path.abspath(os.path.dirname(__file__))    #The path to the main folder

# read the dataset
data = pd.read_csv(os.path.join(general_path,"Btc_small.csv"))
internet_data = pd.read_csv(os.path.join(general_path,"BTC-USD.csv"))
internet_data = internet_data[:len(data)]



data, feature_columns, target_column = feature_engineering(data, internet_data)

# get the train and test frames 
train_size = len(data) - 60
train = data[:train_size]
test = data[train_size:]
# Extract the features and target variable
X_train = train[feature_columns]
Y_train = train[target_column]

x_test = test[feature_columns]
y_test = test[target_column]

X_train_np = X_train.to_numpy()
y_train_np = Y_train.to_numpy()


X_test_np = x_test.to_numpy()
y_test_np = y_test.to_numpy()

##print(X_test_np.shape, y_test_np.shape)

#print(X_train_np.shape, y_train_np.shape)

train_dataset = TensorDataset(torch.tensor(X_train_np, dtype=torch.float),
                              torch.tensor(y_train_np.reshape((-1, 1)), dtype=torch.float))

train_dataloader = DataLoader(train_dataset, batch_size=128)

# for X, y in train_dataloader:
#   print(X.shape, y.shape)
#   break

test_dataset = TensorDataset(torch.tensor(X_test_np, dtype=torch.float), 
                             torch.tensor(y_test_np.reshape((-1, 1)), dtype=torch.float))

test_dataloader = DataLoader(test_dataset, batch_size=64)

# for X, y in test_dataloader:
#   print(X.shape, y.shape)
#   break

from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class LSTM(nn.Module):
  # def __init__(self):
  #   super(NeuralNet, self).__init__()

  #   self.hidden_layer_1 = nn.Linear(26, 64)
  #   self.hidden_activation = nn.ReLU()
    
  #   self.out = nn.Linear(64, 1)
  
  # def forward(self, x):
  #   x = self.hidden_layer_1(x)
  #   x = self.hidden_activation(x)
  #   x = self.out(x)
  #   return x

  def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
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

input_size = 26  
hidden_size = 64  # Increased number of hidden units
num_layers = 2
output_size = 1
batch_size = 64  # Increased batch size
learning_rate = 0.01  # Decreased learning rate


model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
#print(model)


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)


def train(dataloader, model, loss_fn, optimizer):
  model.train()
  train_loss = 0

  for i, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    y_hat = model(X)
    mse = loss_fn(y_hat, y)
    train_loss += mse.item()

    optimizer.zero_grad()
    mse.backward()
    optimizer.step()
  
  num_batches = len(dataloader)
  train_mse = train_loss / num_batches
  rmse = train_mse**(1/2)
  print(f'Train RMSE: {train_mse**(1/2)}')
  return rmse



def test(dataloader, model, loss_fn):
  model.eval()
  test_loss = 0

  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      y_hat = model(X)
      test_loss += loss_fn(y_hat, y).item()
  
  num_batches = len(dataloader)
  test_mse = test_loss / num_batches
  rmse = test_mse**(1/2)
  print(f'Test RMSE: {test_mse**(1/2)}\n')

  return rmse


epochs = 1000

train_rmse_list = []
test_rmse_list = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}:")

    train_rmse = train(train_dataloader, model, loss_fn, optimizer)
    test_rmse = test(test_dataloader, model, loss_fn)

    train_rmse_list.append(train_rmse)
    test_rmse_list.append(test_rmse)

# Plotting the RMSE curve during training
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, epochs + 1), train_rmse_list, label="Train RMSE")
# plt.plot(range(1, epochs + 1), test_rmse_list, label="Test RMSE")
# plt.xlabel("Epoch")
# plt.ylabel("RMSE")
# plt.legend()
# plt.title("Train and Test RMSE during Training")
# plt.show()

# Plotting the actual vs prediction graph
# model.eval()
# with torch.no_grad():
#     test_predictions = model(torch.tensor(X_test_np, dtype=torch.float).to(device))

# plt.figure(figsize=(12, 6))
# plt.plot(y_test_np, label="Actual")
# plt.plot(test_predictions.cpu().numpy(), label="Prediction")
# plt.xlabel("Time")
# plt.ylabel("Target Value")
# plt.legend()
# plt.title("Actual vs Prediction")
# plt.show()







#-----------------------------------------------------------------------------------------

# # scaler = MinMaxScaler(feature_range=(-1,1))
# scaler = scaler.fit(train)

# train = pd.DataFrame(scaler.transform(train), index= train.index, columns= train.columns)
# test = pd.DataFrame(scaler.transform(test), index= test.index, columns= test.columns)

#print(len(test))
# def create_sequences(input_data:pd.DataFrame, target_column, sequence_length):
#     sequences = []
#     data_size = len(input_data)

#     for i in tqdm(range(data_size - sequence_length)):
#         sequence = input_data[i:i+sequence_length]

#         label_position = i + sequence_length
#         label = input_data.iloc[label_position][target_column]

#         sequences.append((sequence, label))

#     return sequences



# sequence_length = 30
# train_sequences = create_sequences(train, 'Open', sequence_length)
# test_sequences = create_sequences(test, 'Open', sequence_length)

# #print(train_sequences)
# #print(len(test_sequences))


# class Bitcoin_Dataset(Dataset):
    
#     def __init__(self, sequences):
#         self.sequences = sequences
    
#     def __len__(self):
#         return len(self.sequences)  #?????
    
#     def __getter__(self,indx):
#         sequence, label = self.sequences[indx]

#         return dict(sequence = torch.Tensor(sequence.to_numpy()),
#                     label = torch.tensor(label).float)
    
# class BTC_Price_Module(pl.LightningDataModule):

#     def __init__(self, train_sequences, test_sequences, batch_size = 8):
#         super().__init__()
#         self.train_sequences = train_sequences
#         self.test_sequences = test_sequences
#         self.batch_size = batch_size

#     def setup(self):
#         self.train_dataset = Bitcoin_Dataset(self.train_sequences)
#         self.test_dataset = Bitcoin_Dataset(self.test_sequences)
    
#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size = self.batch_size,
#                           shuffle = False, num_workers = 1)
    
#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size = self.batch_size,
#                           shuffle = False, num_workers = 1)
    
# BATCH_SIZE = 16
# NUM_EPOCH = 10

# data_module = BTC_Price_Module(train_sequences, test_sequences, batch_size= BATCH_SIZE)
# data_module.setup()

# # Define the LSTM neural network architecture
# class LSTMModel(nn.Module):
#     def __init__(self, n_features, n_hidden, n_layers):
        
#         super().__init__()
        
#         self.hidden_size = n_hidden
#         self.num_layers = n_layers
#         self.lstm = nn.LSTM(input_size = n_features, hidden_size = n_hidden,
#                             num_layers = n_layers, batch_first=True, dropout = 0.2)
#         self.regressor = nn.Linear(n_hidden, 1)

#     def forward(self, x):
        
#         self.lstm.flatten_parameters()

#         _, (hidden, _) = self.lstm(x)
#         out = hidden[-1]

#         return self.regressor(out)
    
# class BTC_Prediction(pl.LightningModule):

#     def __init__(self, n_features:int, n_hidden:int, n_layers:int):

#         super().__init__()

#         self.model = LSTMModel(n_features, n_hidden, n_layers)
#         self.criterion = nn.MSELoss()

#     def forward(self, x, labels= None):

#         output = self.model(x)
#         loss = 0

#         if labels is not None:
#             loss = self.criterion(output, labels.unsqueeze(dim = 1))
        
#         return loss, output
    
#     def training_step(self, batch, batch_indx):
        
#         sequences = batch['sequence']
#         labels = batch['label']

#         loss, outputs = self(sequences, labels)
#         self.log('train_loss', loss, prog_bar = True, logger = True)
#         return loss
    
#     def validation_step(self, batch, batch_indx):
        
#         sequences = batch['sequence']
#         labels = batch['label']

#         loss, outputs = self(sequences, labels)
#         self.log('valid_loss', loss, prog_bar = True, logger = True)
#         return loss
    
#     def test_step(self, batch, batch_indx):
        
#         sequences = batch['sequence']
#         labels = batch['label']

#         loss, outputs = self(sequences, labels)
#         self.log('test_loss', loss, prog_bar = True, logger = True)
#         return loss


#     def optimizer(self):

#         return optim.AdamW(self.parameters(), lr=0.001)
    
# model = BTC_Prediction(n_features=train.shape[1], n_hidden=128, n_layers=2)

# checkpoint = ModelCheckpoint(verbose = True, monitor = 'valid_loss', mode = 'min')
# #logger = TensorBoardLogger('logs', name= 'btc price')
# early_stop = EarlyStopping(patience=2, monitor ='valid_loss')

# trainer = pl.Trainer(callbacks = [early_stop], max_epochs= NUM_EPOCH)
# trainer.fit(model, data_module)

# trained_model = BTC_Prediction.load_from_checkpoint()