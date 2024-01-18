from feature_eng_v2 import feature_engineering
import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings("ignore", category=DataConversionWarning)
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_predict


general_path = os.path.abspath(os.path.dirname(__file__))    #The path to the main folder

# read the dataset
data = pd.read_csv(os.path.join(general_path,"Btc_small.csv"))
internet_data = pd.read_csv(os.path.join(general_path,"BTC-USD.csv"))
internet_data = internet_data[:len(data)]


data, feature_columns, target_column = feature_engineering(data, internet_data)

# get the train and test frames 
train_size = len(data) - 120
valid_lim = train_size + 60

train = data[:train_size]
valid = data[train_size:valid_lim]
# Extract the features and target variable
X_train = train[feature_columns]
Y_train = train[target_column]

x_valid = valid[feature_columns]
y_valid = valid[target_column]

# Define the hyperparameter grid for random search
param_grid = {
      'learning_rate': [0.1, 0.01],
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 6],
    'min_child_weight': [1, 5, 10],
    #'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    #'subsample': [0.7, 0.8, 0.9],
    #'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.001, 0.01, 0.1, 1],
    'reg_lambda': [0, 0.001, 0.01, 0.1, 1],
    'min_child_samples': [1, 5, 10]
  }
xgboost = XGBRegressor()

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(xgboost, param_grid, n_iter=250, cv=5,
                                   random_state=42, verbose=4, scoring="neg_mean_absolute_error")
random_search.fit(X_train, Y_train)

best_params = random_search.best_params_
best_model_lgbm = random_search.best_estimator_

print(best_params)

# Predict on the test data with the best model
best_lgbm = random_search.best_estimator_
pred_lgbm = best_lgbm.predict(x_valid)

# Calculate the root mean squared error
rmse_lgbm = np.sqrt(mean_squared_error(y_valid, pred_lgbm))
print("Root Mean Squared Error (Test - LightGBM):", rmse_lgbm)