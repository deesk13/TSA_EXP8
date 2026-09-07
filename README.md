# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 


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
```
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')


# Read the Walmart dataset from CSV file
data = pd.read_csv('/content/Walmart_Sales.csv')


# Display the shape and first 20 rows
print("Shape of the dataset:", data.shape)

print("First 20 rows of the dataset:")
print(data.head(20))


# Convert Date column to datetime
data['Date'] = pd.to_datetime(
    data['Date'],
    format='%d-%m-%Y'
)


# Aggregate Weekly Sales of all stores by Date
walmart_sales = data.groupby('Date')['Weekly_Sales'].sum()

# Sort the data by Date
walmart_sales = walmart_sales.sort_index()


# Set figure size
plt.rcParams['figure.figsize'] = [10, 6]


# Plot the first 50 values of Weekly Sales
plt.plot(walmart_sales[:50])

plt.title('First 50 Values of Walmart Weekly Sales')
plt.xlabel('Index')
plt.ylabel('Weekly Sales')

plt.grid(True)
plt.show()


# Perform rolling average transformation
# Window size = 5
rolling_mean_5 = walmart_sales.rolling(window=5).mean()


# Display first 10 values
print("First 10 values of the rolling mean (window size 5):")
print(rolling_mean_5.head(10))


# Perform rolling average transformation
# Window size = 10
rolling_mean_10 = walmart_sales.rolling(window=10).mean()


# Plot original data and rolling mean
plt.figure()

plt.plot(
    walmart_sales,
    label='Original Data'
)

plt.plot(
    rolling_mean_10,
    label='Rolling Mean (window=10)'
)

plt.title('Original Walmart Sales and Rolling Mean (window=10)')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')

plt.legend()
plt.grid(True)
plt.show()


# Perform exponential smoothing
alpha = 0.3

exp_smooth = walmart_sales.ewm(
    alpha=alpha,
    adjust=False
).mean()


# Plot original data and exponential smoothing
plt.figure()

plt.plot(
    walmart_sales,
    label='Original Data'
)

plt.plot(
    exp_smooth,
    label='Exponential Smoothing'
)

plt.title('Original Walmart Sales and Exponential Smoothing')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')

plt.legend()
plt.grid(True)
plt.show()


# Implement the Moving Average (MA) Model
# ARIMA order = (0, 0, q)
# p = 0, d = 0, q = 1

q = 1

ma_model = ARIMA(
    walmart_sales,
    order=(0, 0, q)
)

ma_model_fit = ma_model.fit()


# Display model summary
print(ma_model_fit.summary())


# Forecast the next 10 data points
forecast = ma_model_fit.forecast(
    steps=10
)


# Plot original data and forecast
plt.figure()

plt.plot(
    walmart_sales,
    label='Original Data'
)

plt.plot(
    range(len(walmart_sales), len(walmart_sales) + 10),
    forecast,
    label='MA Forecast',
    marker='o'
)

plt.title('Original Walmart Sales and Moving Average Forecast')
plt.xlabel('Index')
plt.ylabel('Weekly Sales')

plt.legend()
plt.grid(True)
plt.show()


# Print forecasted values
print("Forecasted Walmart sales for the next 10 points:")
print(forecast)
```


### OUTPUT:

## Moving Average:
<img width="807" height="651" alt="Screenshot 2026-09-07 162252" src="https://github.com/user-attachments/assets/db8b2d6a-b979-4ef2-be6b-9b0cb2c7316a" />

<img width="720" height="341" alt="image" src="https://github.com/user-attachments/assets/9e972559-5aea-41fe-9381-77303eae932c" />
<img width="917" height="667" alt="image" src="https://github.com/user-attachments/assets/80b123ce-bf6c-45f2-9e6e-296725905dd0" />



Plot Transform Dataset
<img width="922" height="645" alt="image" src="https://github.com/user-attachments/assets/fa6663af-e7ad-4488-b283-1177a00d1548" />



Exponential Smoothing
<img width="961" height="702" alt="image" src="https://github.com/user-attachments/assets/7baf0941-72c4-463b-a4f2-b627e719c6c3" />




### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
