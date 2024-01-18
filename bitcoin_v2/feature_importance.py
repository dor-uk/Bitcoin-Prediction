from feature_eng_v2 import feature_engineering 
import os
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

general_path = os.path.abspath(os.path.dirname(__file__))    #The path to the main folder

# read the dataset
data = pd.read_csv(os.path.join(general_path,"Btc_small.csv"))


data = feature_engineering(data)

#print(data)

feature_columns = ['High', 'Low', 'Close', 'Adj Close', 'Volume',
                   'year', 
                   'month',
                   'day_of_year', 
                   'quarter',
                   'season',
                   'days_remaining',
                   'is_last_days',
                   'months_remaining',
                   'is_last_months',
                   'is_last_months_of2016',
                   'is_second_half_of2016',
                   'month_rad',
                   'sin_month',
                   'Date_numeric',
                   'Date_numeric_sin',
                   'Date_numeric_cos',
                   'is_december',
                   'open_price_lag1',
                   #'open_price_lag2',
                   #'open_price_lag3',
                   #'open_price_lag4',
                   #'open_price_lag5',
                   'open_price_lag9',
                   'rolling_mean',
                   'rolling_std',
                   'rolling_50'
                   ]


target_column = ['Open']


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


xg_boost = XGBRegressor( reg_lambda= 0.7, reg_alpha= 0.7, random_state= 42, n_estimators= 500,  
                        learning_rate= 0.5,   booster= 'gblinear')
xg_boost.fit(X_train, Y_train)

importances = xg_boost.feature_importances_
for feature, importance in zip(X_train, importances):
    print(feature, importance)