from feature_eng_v2 import feature_engineering 
import os
import pandas as pd
import numpy as np 
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


# Set the path to the data files
general_path = os.path.abspath(os.path.dirname(__file__))
data = pd.read_csv(os.path.join(general_path, "Btc_small.csv"))
internet_data = pd.read_csv(os.path.join(general_path, "BTC-USD.csv"))
internet_data = internet_data[:len(data)]
train = internet_data.copy()

data = train['Open']
Date1 = train['Date']
train1 = train[['Date', 'Open']]
# Setting the Date as Index
train1['Date'] = pd.to_datetime(train1['Date'])
train2 = train1.set_index('Date')
train2.sort_index(inplace=True)

# Function to test stationarity of time series
def test_stationarity(x):
    # Determing rolling statistics
    rolmean = x.rolling(window=22, center=False).mean()
    rolstd = x.rolling(window=12, center=False).std()

    # Perform Dickey Fuller test
    result = sm.tsa.adfuller(x)
    print('ADF Stastistic: %f' % result[0])
    print('p-value: %f' % result[1])
    pvalue = result[1]
    for key, value in result[4].items():
        if result[0] > value:
            print("The data is non-stationary")
            break
        else:
            print("The data is stationary")
            break
    print('Critical values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


# Test stationarity of the 'Open' time series
ts = train2['Open']
test_stationarity(ts)

# Convert to log scale
ts_log = np.log(ts)

# Test stationarity of the log-transformed time series
test_stationarity(ts_log)

# Differencing to make the time series stationary
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)

# Test stationarity of the differenced time series
test_stationarity(ts_log_diff)

# plt.plot(ts_log_diff)
# plt.show()
# print(ts_log_diff.info())
#print(ts_log_diff.head())

# Perform feature engineering and split the data into train and test sets
ts_log_diff, feature_columns, target_column = feature_engineering(ts_log_diff, internet_data)

size = int(len(ts_log_diff) - 60)
train_v2, test_v2 = ts_log_diff[:size], ts_log_diff[size:]

# Extract the features and target variable
X_train = train_v2[feature_columns]
Y_train = train_v2[target_column]

X_test = test_v2[feature_columns]
Y_test = test_v2[target_column]

# Initialize the XGBoost regressor
xg_boost = XGBRegressor(
    reg_lambda=0.7,
    reg_alpha=0.7,
    random_state=42,
    n_estimators=500,
    learning_rate=0.5,
    booster='gblinear'
)

# Fit the XGBoost model to the training data
xg_boost.fit(X_train, Y_train)

# Make predictions on the test set
predictions = xg_boost.predict(X_test)
train_predictions = xg_boost.predict(X_train)

# Calculate RMSE (Root Mean Squared Error) on the test set
rmse = mean_squared_error(Y_test, predictions, squared=False)
train_rmse = mean_squared_error(Y_train, train_predictions, squared=False)

print(f'Root Mean Squared Error (Test Set): {rmse}')
print(f'Root Mean Squared Error (Train Set): {train_rmse}')

# Plot the actual vs. predicted values for the test set
plt.figure(figsize=(10, 6))
plt.plot(test_v2.index, np.exp(Y_test), label='Actual', color='blue')
plt.plot(test_v2.index, np.exp(predictions), label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('Actual vs. Predicted Open Prices (Test Set)')
plt.legend()
plt.show()

