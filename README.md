## Developed By: Lakshman
## Register No: 212222240001
# Ex.No: 03   COMPUTE THE AUTO FUNCTION(ACF)
Date: 

### AIM:
To Compute the AutoCorrelation Function (ACF) of the data for the first 35 lags to determine the model
type to fit the data.
### ALGORITHM:
1. Import the necessary packages
2. Find the mean, variance and then implement normalization for the data.
3. Implement the correlation using necessary logic and obtain the results
4. Store the results in an array
5. Represent the result in graphical representation as given below.
### PROGRAM:
~~~python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set seed for reproducibility
np.random.seed(0)

# Load and preprocess data
data = pd.read_csv('/rainfall.csv')
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by='date')
data.set_index('date', inplace=True)
data.dropna(inplace=True)

# Plot the consumption data
plt.figure(figsize=(12, 6))
plt.plot(data['rainfall'], label='Data')
plt.xlabel('date')
plt.ylabel('rainfall')
plt.legend()
plt.title('rainfall Data')
plt.show()

# Split into train and test data
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]
y_train = train_data['rainfall']
y_test = test_data['rainfall']

# Compute and plot ACF for the first 35 lags
plt.figure(figsize=(12, 6))
plot_acf(data['rainfall'], lags=35)
plt.title('ACF of rainfall Data (First 35 Lags)')
plt.show()
# Fit an autoregressive model (AR)
lag_order = 1  # you can adjust based on the ACF plot
data['rainfall'].corr(data['rainfall'].shift(1))
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.api import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error
lag_order = 35 
ar_model = AutoReg(y_train, lags=lag_order)
ar_results = ar_model.fit()

# Predictions
y_pred = ar_results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

# Compute MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
variance = np.var(y_test)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'Variance_testing: {variance:.2f}')
~~~

### OUTPUT:
## VISUAL REPRESENTATION OF DATASET:
<img width="511" alt="Screenshot 2024-09-25 084500" src="https://github.com/user-attachments/assets/baf4055d-1d37-4621-a28c-9c20f6dfba53">

## AUTO CORRELATION:
<img width="281" alt="Screenshot 2024-09-25 084617" src="https://github.com/user-attachments/assets/f4a1dbec-5da4-403e-801f-bc92ea62be51">

## VALUES OF MAE,RMSE,VARIANCE:
<img width="136" alt="Screenshot 2024-09-25 084646" src="https://github.com/user-attachments/assets/dfee2f08-6e4d-435a-b542-7ecd769c2a10">

### RESULT:

Thus we have successfully implemented the auto correlation function in python.
