import os
import pandas as pd
from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np



general_path = os.path.abspath(os.path.dirname(__file__))    #The path to the main folder

# read the dataset
data = pd.read_csv(os.path.join(general_path,"Btc_small.csv"))
# internet_data = pd.read_csv(os.path.join(general_path,"BTC-USD.csv"))
# internet_data = internet_data[:len(data)]
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Calculate numerical representation of 'Date'
data['Date_numeric'] = (data.index - data.index.min()).days

# Function to apply cyclical encoding
def apply_cyclical_encoding(data, column):
    data[column + '_sin'] = np.sin(2 * np.pi * data[column] / data[column].max())
    data[column + '_cos'] = np.cos(2 * np.pi * data[column] / data[column].max())
    return data

data = apply_cyclical_encoding(data, 'Date_numeric')

#print(data)


# get the train and test frames 
train_size = len(data) - 60
valid_lim = train_size

train = data[:train_size]
test = data[valid_lim:]

X_train = train[['Date_numeric_sin', 'Date_numeric_cos']]
Y_train = train['Open']

x_test = test[['Date_numeric_sin', 'Date_numeric_cos']]
y_test = test['Open']

#print(Y_train)


'''reg_lambda= 0.1, reg_alpha= 0.0, random_state= 42, n_estimators= 500, max_depth= 3, learning_rate= 0.1'''

xg_boost = XGBRegressor( )
#linear_reg = LinearRegression()
#rf = RandomForestRegressor()

xg_boost.fit(X_train, Y_train)
pred_xg = xg_boost.predict(x_test)
pred_train_xg = xg_boost.predict(X_train)
rmse_xg = mean_squared_error(y_test, pred_xg, squared=False)
rmse_train_xg = mean_squared_error(Y_train, pred_train_xg, squared=False)

# Linear Regression and SVR

fig, ax = plt.subplots(figsize=(15, 8))


#plt.figure(figsize=(10, 8))
ax.plot(y_test.index, y_test, label='Actual')
ax.plot(y_test.index, pred_xg, label='XGBoost')
ax.set_xlabel('Date')
ax.set_ylabel('Bitcoin Open Price')
ax.set_title('XGBoost Prediction Without Feature Engineering')
ax.legend()



plt.show()

naive_test = mean_squared_error(y_test,y_test.shift(1).bfill(), squared=False)
print("Naive Test:", naive_test)
print("***********************************************************************************")
#print("-----------------------------------------------------------------------------------")
print("Root Mean Squared Error (Train - XGBoost):", rmse_train_xg)
print("Root Mean Squared Error (XGBoost):", rmse_xg)