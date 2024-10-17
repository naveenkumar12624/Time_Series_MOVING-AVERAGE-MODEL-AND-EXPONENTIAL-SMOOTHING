
# Ex.No: 08 - MOVING AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 
### Developed by: Naveen Kumar S
### Reg no: 212221240033

### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Suppress warnings
warnings.filterwarnings('ignore')

# Read the dataset (adjust the path accordingly)
data = pd.read_csv('C:/Users/lenovo/Downloads/archive (2)/NVIDIA/NvidiaStockPrice.csv')

# Convert 'Date' to datetime format and set it as the index
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data.set_index('Date', inplace=True)

# Focus on the 'Adj Close' column
adj_close_data = data[['Adj Close']]

# Filter data to start from the year 2018
adj_close_data = adj_close_data[adj_close_data.index >= '2020-01-01']

# Display the shape and the first 5 rows of the dataset
print("Shape of the dataset:", adj_close_data.shape)
print("First 5 rows of the dataset:")
print(adj_close_data.head())

# Plot Original Dataset (Adj Close Data)
plt.figure(figsize=(12, 6))
plt.plot(adj_close_data['Adj Close'], label='Original Adj Close Data', color='blue')
plt.title('Original Adjusted Close Prices')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (USD)')
plt.legend()
plt.grid()
plt.show()

# Moving Average
# Perform rolling average transformation with a window size of 10
rolling_mean_10 = adj_close_data['Adj Close'].rolling(window=10).mean()

# Plot Moving Average
plt.figure(figsize=(12, 6))
plt.plot(adj_close_data['Adj Close'], label='Original Adj Close Data', color='blue')
plt.plot(rolling_mean_10, label='Moving Average (window=10)', color='orange')
plt.title('Moving Average of Adjusted Close Prices')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (USD)')
plt.legend()
plt.grid()
plt.show()

# Exponential Smoothing
model = ExponentialSmoothing(adj_close_data['Adj Close'], trend='add', seasonal=None)
model_fit = model.fit()

# Number of years to predict
x_years = 2  # Change this value to predict for a different number of years
n_periods = x_years * 252  # Assuming trading days per year is approximately 252

# Make predictions for the next n_periods
predictions = model_fit.predict(start=len(adj_close_data), end=len(adj_close_data) + n_periods - 1)

# Create a date range for the predictions
last_date = adj_close_data.index[-1]
prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_periods, freq='B')  # 'B' for business days

# Plot the original data and Exponential Smoothing predictions
plt.figure(figsize=(12, 6))
plt.plot(adj_close_data['Adj Close'], label='Original Adj Close Data', color='blue')
plt.plot(prediction_dates, predictions, label='Exponential Smoothing Forecast', color='orange')
plt.title(f'Exponential Smoothing Predictions for Adjusted Close Prices (Next {x_years} Years)')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (USD)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/94bd403a-a02c-4928-93e7-bfadf3553319)

### Adjusted Close Price (USD):
![download](https://github.com/user-attachments/assets/fe07aadd-f51f-4a59-abb2-353178ff46e6)

### Moving Average
![download](https://github.com/user-attachments/assets/2f666614-334d-4cb6-a43d-9f9de3d49701)

### Exponential Smoothing
![download](https://github.com/user-attachments/assets/4871c6e3-952f-4f72-b718-ad63566a64dd)


### RESULT:
Thus, implemention of the Moving Average Model and Exponential smoothing using python is successful.
