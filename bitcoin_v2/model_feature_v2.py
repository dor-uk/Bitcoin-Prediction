from feature_eng_v2 import feature_engineering 
import os
import pandas as pd
from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


general_path = os.path.abspath(os.path.dirname(__file__))    #The path to the main folder

# read the dataset
data = pd.read_csv(os.path.join(general_path,"Btc_small.csv"))
internet_data = pd.read_csv(os.path.join(general_path,"BTC-USD.csv"))
internet_data = internet_data[:len(data)]


data, feature_columns, target_column = feature_engineering(data, internet_data)

#print(data)


# get the train and test frames 
train_size = len(data) - 60
valid_lim = train_size

train = data[:train_size]
test = data[valid_lim:]
# Extract the features and target variable
X_train = train[feature_columns]
Y_train = train[target_column]

x_test = test[feature_columns]
y_test = test[target_column]

#print(Y_train)


'''reg_lambda= 0.1, reg_alpha= 0.0, random_state= 42, n_estimators= 500, max_depth= 3, learning_rate= 0.1'''

xg_boost = XGBRegressor( reg_lambda= 0.7, reg_alpha= 0.7, random_state= 42, n_estimators= 500,  
                        learning_rate= 0.5,   booster= 'gblinear')
#linear_reg = LinearRegression()
#rf = RandomForestRegressor()

xg_boost.fit(X_train, Y_train)
pred_xg = xg_boost.predict(x_test)
pred_train_xg = xg_boost.predict(X_train)
rmse_xg = mean_squared_error(y_test, pred_xg, squared=False)
rmse_train_xg = mean_squared_error(Y_train, pred_train_xg, squared=False)
r2_score_xg = r2_score(y_test, pred_xg)
r2_score_train_xg = r2_score(Y_train, pred_train_xg)

# Linear Regression and SVR


plt.figure(figsize=(10, 8))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, pred_xg, label='XGBoost')
plt.xlabel('Date')
plt.ylabel('Bitcoin Open Price')
plt.title('XGBoost Prediction')
plt.legend()

plt.figure(figsize=(10, 8))
plt.plot(Y_train.index, Y_train, label='Actual')
plt.plot(Y_train.index, pred_train_xg, label='XGBoost')
plt.xlabel('Date')
plt.ylabel('Bitcoin Open Price')
plt.title('XGBoost Train Set Prediction')
plt.legend()

plt.show()


naive_test = mean_squared_error(y_test,y_test.shift(1).bfill(), squared=False)
print("Naive Test:", naive_test)
#print("***********************************************************************************")
print("-----------------------------------------------------------------------------------")
print("Root Mean Squared Error (Train - XGBoost):", rmse_train_xg)
print("Root Mean Squared Error (XGBoost):", rmse_xg)
print("R2 Score (Train - XGBoost):", r2_score_train_xg)
print("R2 Score (XGBoost):", r2_score_xg)
