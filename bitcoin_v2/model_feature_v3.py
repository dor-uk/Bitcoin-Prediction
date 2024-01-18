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

#print(len(y_test))
'reg_lambda= 0.7, reg_alpha= 0.7, random_state= 42, n_estimators= 500,  learning_rate= 0.5,   booster= gblinear' 
xg_boost = XGBRegressor(reg_lambda= 0.5, reg_alpha= 1, n_estimators= 500, learning_rate= 0.3,
                        booster= 'gblinear', random_state= 42)

xg_boost.fit(X_train, Y_train)
pred_train_xg = xg_boost.predict(X_train)
pred_valid_xg = xg_boost.predict(X_val)
pred_xg = xg_boost.predict(x_test)

rmse_xg = mean_squared_error(y_test, pred_xg, squared=False)
rmse_train_xg = mean_squared_error(Y_train, pred_train_xg, squared=False)
rmse_val_xg = mean_squared_error(y_val, pred_valid_xg, squared=False)

r2_score_xg = r2_score(y_test, pred_xg)
r2_score_train_xg = r2_score(Y_train, pred_train_xg)
r2_score_valid_xg = r2_score(y_val, pred_valid_xg)

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test.index, y=y_test["Open"], name="Actual_test"))
fig.add_trace(go.Scatter(x=y_test.index, y=pred_xg, name="XGBoost Test"))
fig.add_trace(go.Scatter(x=y_val.index, y=y_val["Open"], name="Actual_valid", line= dict(color= 'black')))
fig.add_trace(go.Scatter(x=y_val.index, y=pred_valid_xg, name="XGBoost valid", line= dict(color= 'magenta')))
fig.add_trace(go.Scatter(x=Y_train.index, y=Y_train["Open"], name="Actual_train"))
fig.add_trace(go.Scatter(x=Y_train.index, y=pred_train_xg, name="XGBoost train"))


fig.update_layout(height=750)
fig.show()


naive_test = mean_squared_error(y_test,y_test.shift(1).bfill(), squared=False)
print("Naive Test:", naive_test)
#print("***********************************************************************************")
print("-----------------------------------------------------------------------------------")
print("Root Mean Squared Error (Train - XGBoost):", rmse_train_xg)
print("Root Mean Squared Error (Valid - XGBoost):", rmse_val_xg)
print("Root Mean Squared Error (XGBoost):", rmse_xg)
print("-----------------------------------------------------------------------------------")
print("R2 Score (Train - XGBoost):", r2_score_train_xg)
print("R2 Score (Valid - XGBoost):", r2_score_valid_xg)
print("R2 Score (XGBoost):", r2_score_xg)
