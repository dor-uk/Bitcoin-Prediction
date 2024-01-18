import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import warnings
from statsmodels.tsa.arima.model import ARIMA
from feature_eng_v2 import feature_engineering 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

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

   #  # Plot rolling statistics:
   #  orig = plt.plot(x, color='blue', label='Original')
   #  mean = plt.plot(rolmean, color='red', label='Rolling Mean')
   #  std = plt.plot(rolstd, color='black', label='Rolling Std')
   #  plt.legend(loc='best')
   #  plt.title('Rolling Mean & Standard Deviation')
   #  plt.show(block=False)

    # Perform Dickey Fuller test
    result = sm.tsa.adfuller(x)
    print('ADF Stastistic: %f' % result[0])
    print('p-value: %f' % result[1])
    pvalue = result[1]
    for key, value in result[4].items():
        if result[0] > value:
            print("The graph is non-stationary")
            break
        else:
            print("The graph is stationary")
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

# Train-test-validation split
train_size = int(len(ts_log) -120)
val_size = 60
test_size = len(ts_log) - train_size - val_size

train_arima, val_arima, test_arima = ts_log[:train_size], ts_log[train_size:train_size + val_size], ts_log[train_size + val_size:]

# ARIMAX model

train, feature_columns, target_column = feature_engineering(train, internet_data)

exog_train = train[feature_columns]  # Use some features as exogenous variables
exog_train = exog_train.to_numpy()

history = train_arima.tolist()
predictions_train = list()
predictions_val = list()
predictions_test = list()
originals_train = list()
originals_val = list()
originals_test = list()
error_list_train = list()
error_list_val = list()
error_list_test = list()

print('Printing Predicted vs Expected Values (ARIMAX)...\n')
#print(history, len(history), len(exog_train))

for t in range(len(train_arima)):
    model = ARIMA(history, order=(2, 1, 0), exog=exog_train[:len(history)])
    model_fit = model.fit()

    # Forecast using the last value from the exogenous variables
    exog_forecast = exog_train[t:t+1]  # Slice the relevant exog values for the forecast

    output = model_fit.forecast(steps=1, exog=exog_forecast)

    
    pred_value = np.exp(output[0])  # Extracting the forecasted value from the output array
    original_value = np.exp(train_arima[t])
    
    history.append(train_arima[t])  # Update the history with the original value (not the log transformed)
    
    # Calculating the error
    error = ((abs(pred_value - original_value)) / original_value) * 100
    error_list_val.append(error)
    print('train --> step = %f, predicted = %f, expected = %f, error = %f%%' % (t, pred_value, original_value, error))
    
    predictions_train.append(float(pred_value))
    originals_train.append(float(original_value))





for t in range(len(val_arima)):
    model = ARIMA(history, order=(2, 1, 0), exog=exog_train[:len(history)])
    model_fit = model.fit()

    # Forecast using the last value from the exogenous variables
    exog_forecast = exog_train[t:t+1]  # Slice the relevant exog values for the forecast

    output = model_fit.forecast(steps=1, exog=exog_forecast)

    
    pred_value = np.exp(output[0])  # Extracting the forecasted value from the output array
    original_value = np.exp(val_arima[t])
    
    history.append(val_arima[t])  # Update the history with the original value (not the log transformed)
    
    # Calculating the error
    error = ((abs(pred_value - original_value)) / original_value) * 100
    error_list_val.append(error)
    print('valid --> step = %f, predicted = %f, expected = %f, error = %f%%' % (t, pred_value, original_value, error))
    
    predictions_val.append(float(pred_value))
    originals_val.append(float(original_value))

for t in range(len(test_arima)):
    model = ARIMA(history, order=(2, 1, 0), exog=exog_train[:len(history)])
    model_fit = model.fit()

    # Forecast using the last value from the exogenous variables
    exog_forecast = exog_train[t:t+1]  # Slice the relevant exog values for the forecast

    output = model_fit.forecast(steps=1, exog=exog_forecast)

    
    pred_value = np.exp(output[0])  # Extracting the forecasted value from the output array
    original_value = np.exp(test_arima[t])
    
    history.append(test_arima[t])  # Update the history with the original value (not the log transformed)
    
    # Calculating the error
    error = ((abs(pred_value - original_value)) / original_value) * 100
    error_list_test.append(error)
    print('test --> step = %f, predicted = %f, expected = %f, error = %f%%' % (t, pred_value, original_value, error))
    
    predictions_test.append(float(pred_value))
    originals_test.append(float(original_value))


# Calculate mean error on the test set   
train_mean_error = sum(error_list_train) / float(len(error_list_train))
val_mean_error = sum(error_list_val) / float(len(error_list_val))
test_mean_error = sum(error_list_test) / float(len(error_list_test))
print('\nMean Error in Predicting Train Case Articles (ARIMAX): %f%%' % train_mean_error)
print('\nMean Error in Predicting Valid Case Articles (ARIMAX): %f%%' % val_mean_error)
print('\nMean Error in Predicting Test Case Articles (ARIMAX): %f%%' % test_mean_error)
rmse_arimax_train = mean_squared_error(originals_train, predictions_train, squared=False)
rmse_arimax_val = mean_squared_error(originals_val, predictions_val, squared=False)
rmse_arimax_test = mean_squared_error(originals_test, predictions_test, squared=False)


print("Root Mean Squared Error (ARIMAX):", rmse_arimax_train)
print("Root Mean Squared Error (ARIMAX):", rmse_arimax_val)
print("Root Mean Squared Error (ARIMAX):", rmse_arimax_test)

# plot
fig, ax = plt.subplots(figsize=(15, 8))

# Add data series to the plot
ax.plot(train_arima.index, predictions_train, label="Train Predictions", color='dodgerblue')
ax.plot(val_arima.index, predictions_val, label="Validation Predictions", color='cyan')
ax.plot(test_arima.index, predictions_test, label="Test Predictions", color='orange')
ax.plot(train_arima.index, originals_train, label="Train Target", color='brown')
ax.plot(val_arima.index, originals_val, label="Validation Target", color='black')
ax.plot(test_arima.index, originals_test, label="Test Target", color='green')

# Customize the plot
ax.set_title('Prediction vs Target')
ax.set_xlabel('Date')
ax.set_ylabel('Open Price')
ax.legend()

# Display the plot

plt.show()
