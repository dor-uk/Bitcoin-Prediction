from feature_eng_v2 import lightgbm_engineering 
import os
import pandas as pd
from xgboost import XGBRegressor
#import lightgbm
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np



general_path = os.path.abspath(os.path.dirname(__file__))    #The path to the main folder

# read the dataset
data = pd.read_csv(os.path.join(general_path,"Btc_small.csv"))
internet_data = pd.read_csv(os.path.join(general_path,"BTC-USD.csv"))
internet_data = internet_data[:len(data)]


data, feature_columns, target_column = lightgbm_engineering(data, internet_data)

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


light_gbm = LGBMRegressor(subsample= 0.7, n_estimators= 150, min_child_samples= 3, max_depth= 11, learning_rate= 0.01, colsample_bytree= 0.9)


light_gbm.fit(X_train, Y_train)
pred_lgbm = light_gbm.predict(x_test)
pred_train_lgbm = light_gbm.predict(X_train)
rmse_lgbm = mean_squared_error(y_test, pred_lgbm, squared=False)
rmse_train_lgbm = mean_squared_error(Y_train, pred_train_lgbm, squared=False)




plt.figure(figsize=(10, 8))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, pred_lgbm, label='LightGBM')
plt.xlabel('Date')
plt.ylabel('Bitcoin Open Price')
plt.title('LGBM Prediction')
plt.legend()

plt.figure(figsize=(10, 8))
plt.plot(Y_train.index, Y_train, label='Actual')
plt.plot(Y_train.index, pred_train_lgbm, label='LightGBM')
plt.xlabel('Date')
plt.ylabel('Bitcoin Open Price')
plt.title('LGBM Train Set Prediction')
plt.legend()

plt.show()
naive_test = mean_squared_error(y_test,y_test.shift(1).bfill(), squared=False)
print("Naive Test:", naive_test)
print("***********************************************************************************")

#print("-----------------------------------------------------------------------------------")
print("Root Mean Squared Error (Train - XGBoost):", rmse_train_lgbm)
print("Root Mean Squared Error (XGBoost):", rmse_lgbm)
