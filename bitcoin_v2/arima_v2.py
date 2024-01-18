import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

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

# Train-test split
size = int(len(ts_log) - 60)
train_arima, test_arima = ts_log[0:size], ts_log[size:]

# ARIMA model
history = [x for x in train_arima]
predictions = list()
originals = list()
error_list = list()

print('Printing Predicted vs Expected Values...\n')

# Fit ARIMA model and make predictions on the test set
for t in range(len(test_arima)):
    model = ARIMA(history, order=(2, 1, 0))
    model_fit = model.fit()
    
    output = model_fit.forecast(steps=1)
    
    pred_value = np.exp(output[0])  # Extracting the forecasted value from the output array
    original_value = np.exp(test_arima[t])
    
    history.append(test_arima[t])  # Update the history with the original value (not the log transformed)
    
    # Calculating the error
    error = ((abs(pred_value - original_value)) / original_value) * 100
    error_list.append(error)
    print('predicted = %f, expected = %f, error = %f%%' % (pred_value, original_value, error))
    
    predictions.append(float(pred_value))
    originals.append(float(original_value))

# Calculate mean error on the test set   
mean_error = sum(error_list) / float(len(error_list))
print('\nMean Error in Predicting Test Case Articles: %f%%' % mean_error)

# naive_test = mean_squared_error(originals, originals.shift(1).bfill(), squared=False)
# print("Naive Test:", naive_test)
# print("***********************************************************************************")
rmse_arima = mean_squared_error(originals, predictions, squared=False)
print("Root Mean Squared Error (ARIMA):", rmse_arima)


import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_arima.index[1:], y=originals, name="Actual_test"))
fig.add_trace(go.Scatter(x=test_arima.index[1:], y=predictions, name="Arimax Test Pred"))

fig.update_layout(height=750)
fig.show()



#print(test_arima)
# Plotting the predictions
# plt.figure(figsize=(8, 6))
# test_day = [t for t in range(len(test_arima))]
# labels = {'Original', 'Predicted'}
# plt.plot(test_arima.index, predictions, color='magenta')
# plt.plot(test_arima.index, originals, color='black')
# plt.title('Expected Vs Predicted Views Forecasting (ARIMA)')
# plt.xlabel('Day')
# plt.ylabel('Open Price')
# plt.legend(labels)
# plt.show()