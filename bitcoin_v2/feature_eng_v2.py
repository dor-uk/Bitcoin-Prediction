import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import datetime 
from ta.trend import MACD
import pandas_ta as ta


# general_path = os.path.abspath(os.path.dirname(__file__))    #The path to the main folder

# # read the dataset
# data = pd.read_csv(os.path.join(general_path,"Btc_small.csv"))
# internet_data = pd.read_csv(os.path.join(general_path,"BTC-USD.csv"))
# internet_data = internet_data[:len(data)]

#print(data.info())
#print(internet_data.info())

def feature_engineering(data, internet_data):
    
    # if not isinstance(data.index, datetime.date):
    data['Date'] = pd.to_datetime(data['Date'])
    internet_data['Date'] = pd.to_datetime(internet_data['Date'])

    data = internet_data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    # Set 'Date' column as the index
    data.set_index('Date', inplace=True)


    # Extract components from the date column
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['day_of_year'] = data.index.dayofyear
    data['quarter'] = data.index.quarter
    data['season'] = data['month'] % 12 // 3 + 1



    # Trigonometric transformation for 'quarter' feature
    data['quarter_sin'] = np.sin(2 * np.pi * data['quarter'] / 4)
    data['quarter_cos'] = np.cos(2 * np.pi * data['quarter'] / 4)

    # Trigonometric transformation for 'season' feature
    data['season_sin'] = np.sin(2 * np.pi * data['season'] / 4)
    data['season_cos'] = np.cos(2 * np.pi * data['season'] / 4)

    # Calculate the number of days remaining in the year
    data['days_remaining'] = 365 - data['day_of_year']
    # Define a threshold for the number of days considered as "last days"
    last_days_threshold = 10
    # Create a binary indicator for last days of the year
    data['is_last_days'] = (data['days_remaining'] <= last_days_threshold).astype(int)

    data['months_remaining'] = 12 - data['month']
    last_months_threshold = 3
    data['is_last_months'] = (data['months_remaining'] <= last_days_threshold).astype(int)

    # Create a binary indicator for last months of 2016
    is_last_months_of2016 = ((data['year'] == 2016) & (data['months_remaining'] <= last_months_threshold)).astype(int)
    data['is_last_months_of2016'] = is_last_months_of2016

    is_second_half_of2016 = ((data['year'] == 2016) & (data['months_remaining'] <= 7)).astype(int)
    data['is_second_half_of2016'] = is_second_half_of2016



    # derive the trigonometric cycles from the components of date
    data['month_rad'] = 2 * np.pi * (data['month'] - 1) / 12
    data['day_rad'] = 2 * np.pi * (data['day'] - 1) / 31

    # Apply sine and cosine transformations
    data['sin_month'] = np.sin(data['month_rad'])
    data['cos_month'] = np.cos(data['month_rad'])
    data['sin_day'] = np.sin(data['day_rad'])
    data['cos_day'] = np.cos(data['day_rad'])


    # Calculate numerical representation of 'Date'
    data['Date_numeric'] = (data.index - data.index.min()).days

    # Function to apply cyclical encoding
    def apply_cyclical_encoding(data, column):
        data[column + '_sin'] = np.sin(2 * np.pi * data[column] / data[column].max())
        data[column + '_cos'] = np.cos(2 * np.pi * data[column] / data[column].max())
        return data

    data = apply_cyclical_encoding(data, 'Date_numeric')


    # add new features
    data['is_saturday'] = (data.index.weekday == 5).astype(int)      
    data['is_monday'] = (data.index.weekday == 0).astype(int)         
    data['is_tuesday'] = (data.index.weekday == 1).astype(int)
    data['is_thursday'] = (data.index.weekday == 3).astype(int)
    data['is_sunday'] = (data.index.weekday == 6).astype(int)         
    data['is_december'] = (data['month'] == 12).astype(int)           
    data['is_july'] = (data['month'] == 7).astype(int)  
    data['open_price_lag1'] = data['Open'].shift(1)                   # from the EDA we know that lag1 is important
    data['open_price_lag9'] = data['Open'].shift(9)                   # from the EDA we know that lag9 is important
    data['Close'] = data['Close'].shift(2)
    data['High'] = data['High'].shift(1)
    data['Low'] = data['Low'].shift(1)
    data['Volume'] = data['Volume'].shift(1)
    data['Adj Close'] = data['Adj Close'].shift(2)
    

    interval = len(data) - 60
    #Apply rolling window statistics to 'Open' column
    data['rolling_mean'] = data[:interval]['Open'].rolling(window=7,closed = 'left').mean() 
    data['rolling_std'] = data[:interval]['Open'].rolling(window=7, closed = 'left').std()
    data['rolling_50'] = data[:interval]['Open'].rolling(window=50,closed = 'left').mean() 


    # Function to calculate MACD
    def calculate_macd(data):
        
        data['ema12'] = data['Adj Close'].ewm(span=12).mean()
        data['ema26'] = data['Adj Close'].ewm(span=26).mean()

        data['macd'] = data['ema12'] - data['ema26']

        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']

        return data

    data = calculate_macd(data)
    
    
    # Function to calculate RSI
    def calculate_rsi(data):
        change = data["Open"].diff()
        change.dropna(inplace=True)

        change_up = change.copy()
        change_down = change.copy()

        # 
        change_up[change_up<0] = 0
        change_down[change_down>0] = 0

        # Verify that we did not make any mistakes
        change.equals(change_up+change_down)

        # Calculate the rolling average of average up and average down
        avg_up = change_up.rolling(14).mean()
        avg_down = change_down.rolling(14).mean().abs()

        rsi = 100 * avg_up / (avg_up + avg_down)

        # Take a look at the 20 oldest datapoints
        rsi.head(20)

        data['RSI'] = rsi

        return data
    
    data = calculate_rsi(data)

    data = data.drop(columns=['Adj Close'])

    # Mean Imputation
    data.fillna(data.mean(),inplace=True)
    data.isnull().sum()

    feature_columns = ['High', 'Low', 'Close', 'Volume',
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
                   #'open_price_lag9',
                   'rolling_mean',
                   'rolling_std',
                   'rolling_50',
                   'ema12',
                   'ema26',
                   'macd',
                   'RSI'
                   ]


    target_column = ['Open']

    return data, feature_columns, target_column


def neural_engineering(data, internet_data):
    
    # if not isinstance(data.index, datetime.date):
    data['Date'] = pd.to_datetime(data['Date'])
    internet_data['Date'] = pd.to_datetime(internet_data['Date'])

    data = internet_data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    # Set 'Date' column as the index
    data.set_index('Date', inplace=True)


    # Extract components from the date column
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['day_of_year'] = data.index.dayofyear
    data['quarter'] = data.index.quarter
    data['season'] = data['month'] % 12 // 3 + 1

    # Trigonometric transformation for 'quarter' feature
    data['quarter_sin'] = np.sin(2 * np.pi * data['quarter'] / 4)
    data['quarter_cos'] = np.cos(2 * np.pi * data['quarter'] / 4)

    # Trigonometric transformation for 'season' feature
    data['season_sin'] = np.sin(2 * np.pi * data['season'] / 4)
    data['season_cos'] = np.cos(2 * np.pi * data['season'] / 4)


    # Calculate the number of days remaining in the year
    data['days_remaining'] = 365 - data['day_of_year']
    # Define a threshold for the number of days considered as "last days"
    last_days_threshold = 10
    # Create a binary indicator for last days of the year
    data['is_last_days'] = (data['days_remaining'] <= last_days_threshold).astype(int)

    data['months_remaining'] = 12 - data['month']
    last_months_threshold = 3
    data['is_last_months'] = (data['months_remaining'] <= last_days_threshold).astype(int)

    # Create a binary indicator for last months of 2016
    is_last_months_of2016 = ((data['year'] == 2016) & (data['months_remaining'] <= last_months_threshold)).astype(int)
    data['is_last_months_of2016'] = is_last_months_of2016

    is_second_half_of2016 = ((data['year'] == 2016) & (data['months_remaining'] <= 7)).astype(int)
    data['is_second_half_of2016'] = is_second_half_of2016



    # derive the trigonometric cycles from the components of date
    data['month_rad'] = 2 * np.pi * (data['month'] - 1) / 12
    data['day_rad'] = 2 * np.pi * (data['day'] - 1) / 31

    # Apply sine and cosine transformations
    data['sin_month'] = np.sin(data['month_rad'])
    data['cos_month'] = np.cos(data['month_rad'])
    data['sin_day'] = np.sin(data['day_rad'])
    data['cos_day'] = np.cos(data['day_rad'])


    # Calculate numerical representation of 'Date'
    data['Date_numeric'] = (data.index - data.index.min()).days

    # Function to apply cyclical encoding
    def apply_cyclical_encoding(data, column):
        data[column + '_sin'] = np.sin(2 * np.pi * data[column] / data[column].max())
        data[column + '_cos'] = np.cos(2 * np.pi * data[column] / data[column].max())
        return data

    data = apply_cyclical_encoding(data, 'Date_numeric')


    # add new features
    data['is_saturday'] = (data.index.weekday == 5).astype(int)       # price is high
    data['is_monday'] = (data.index.weekday == 0).astype(int)         # price is low
    data['is_tuesday'] = (data.index.weekday == 1).astype(int)
    data['is_thursday'] = (data.index.weekday == 3).astype(int)
    data['is_sunday'] = (data.index.weekday == 6).astype(int)         # price is low
    data['is_december'] = (data['month'] == 12).astype(int)           # price is high
    data['is_july'] = (data['month'] == 7).astype(int)  
    #data['is_2017'] = (data['year'] == 2017).astype(int)              # price is high
    data['open_price_lag1'] = data['Open'].shift(1)                   # from the EDA we know that lag1 is important
    data['open_price_lag9'] = data['Open'].shift(9)                   # from the EDA we know that lag9 is important
    data['Close'] = data['Close'].shift(2)
    data['High'] = data['High'].shift(2)
    data['Low'] = data['Low'].shift(2)
    data['Volume'] = data['Volume'].shift(2)
    
    


    interval = len(data) - 90
    #Apply rolling window statistics to 'Open' column
    data['rolling_mean'] = data[:interval]['Open'].rolling(window=7,closed = 'left').mean() 
    data['rolling_std'] = data[:interval]['Open'].rolling(window=7, closed = 'left').std()
    data['rolling_50'] = data[:interval]['Open'].rolling(window=50,closed = 'left').mean() 
    
# Function to calculate MACD
    def calculate_macd(data):
        
        data['ema12'] = data['Adj Close'].ewm(span=12).mean()
        data['ema26'] = data['Adj Close'].ewm(span=26).mean()

        data['macd'] = data['ema12'] - data['ema26']

        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']

        return data

    data = calculate_macd(data)
    
    
    # Function to calculate RSI
    def calculate_rsi(data):
        change = data["Open"].diff()
        change.dropna(inplace=True)

        change_up = change.copy()
        change_down = change.copy()

        # 
        change_up[change_up<0] = 0
        change_down[change_down>0] = 0

        # Verify that we did not make any mistakes
        change.equals(change_up+change_down)

        # Calculate the rolling average of average up and average down
        avg_up = change_up.rolling(14).mean()
        avg_down = change_down.rolling(14).mean().abs()

        rsi = 100 * avg_up / (avg_up + avg_down)

        # Take a look at the 20 oldest datapoints
        rsi.head(20)

        data['RSI'] = rsi

        return data
    
    data = calculate_rsi(data)

    data = data.drop(columns=['Adj Close'])

    # Mean Imputation
    data.fillna(data.mean(),inplace=True)
    data.isnull().sum()

    feature_columns = ['High', 'Low', 'Close', 'Volume',
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
                   #'open_price_lag9',
                   'rolling_mean',
                   'rolling_std',
                   'rolling_50',
                   'ema12',
                   'ema26',
                   'macd',
                   'RSI'
                   ]

    target_column = ['Open']

    return data, feature_columns, target_column


#data, feature_columns, target_column = feature_engineering(data,internet_data)
# analyze the correlation between variables, we have only one variable so the corelation will be 1
def corr_plot(data):
    correlation = data.corr(method='pearson', numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')

    plt.show()
#corr_plot(data)