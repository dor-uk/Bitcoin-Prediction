from feature_eng_v2 import feature_engineering 
import os
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


general_path = os.path.abspath(os.path.dirname(__file__))    #The path to the main folder

# read the dataset
data = pd.read_csv(os.path.join(general_path,"Btc_small.csv"))
internet_data = pd.read_csv(os.path.join(general_path,"BTC-USD.csv"))
data = internet_data[:len(data)]
data = feature_engineering(data)


opendf = data[['Open']]
#opendf = opendf[train_size:]
open_stock = opendf.copy()

#del opendf['Date']

from copy import deepcopy as dc
def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    
    #df.set_index('Date', inplace=True)
    
    for i in range(1, n_steps+1):
        df[f'Open(t-{i})'] = df['Open'].shift(i)
        
    df.dropna(inplace=True)
    
    return df

lookback = 7
shifted_df = prepare_dataframe_for_lstm(opendf, lookback)
shifted_df_as_np = shifted_df.to_numpy()
#print(shifted_df)
#print(shifted_df_as_np.shape)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

#print(shifted_df_as_np.shape)
#print(shifted_df_as_np)


X = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]


X = dc(np.flip(X, axis=1))



split_index = len(opendf) - 60


X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]



#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# X_train = X_train.reshape((-1, lookback, 1))
# X_test = X_test.reshape((-1, lookback, 1))

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

#print(y_train.shape, y_test.shape)


xg_boost = XGBRegressor()

xg_boost.fit(X_train, y_train)
pred_xg = xg_boost.predict(X_test)

test_predictions = pred_xg.flatten()

dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)

test_predictions = dc(dummies[:, 0])


dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = y_test.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_test = dc(dummies[:, 0])



plt.plot(new_y_test, label='Actual Open')
plt.plot(test_predictions, label='Predicted Open')
plt.xlabel('Day')
plt.ylabel('Open')
plt.legend()
plt.show()