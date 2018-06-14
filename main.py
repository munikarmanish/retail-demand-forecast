import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil import parser as dtparser
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

# Load CSV
print('Loading CSV...')
csv_data = pd.read_csv('SalesData.csv',
                       index_col='Day',
                       parse_dates=['Day', 'Fiscal Week'],
                       date_parser=lambda s: dtparser.parse(s).date())

# Filter Class 10 Sales Dollars
class10_sales_data = csv_data.query('Class==10').SalesD.astype('float')

# Fill in missing values (some dates are missing in the index)
start_date = class10_sales_data.index[0].date()
end_date = class10_sales_data.index[-1].date()
date_range = pd.date_range(start_date, end_date)
X = class10_sales_data.reindex(date_range, fill_value=0).values

# Divide into train and test
train = X[:600]
test = X[600:]
train.size, test.size

# Forecast using ARIMA
history = list(train)
predicted = []
for i in range(len(test)):
    if (i+1) % 10 == 0:
        print(" Forecasting #{}".format(i+1))
    predicted.append(ARIMA(history, order=(6, 1, 0)).fit(disp=0).forecast()[0])
    history.append(test[i])

# Let's see the predictions compared to actual values
plt.plot(test, label='Actual')
plt.plot(predicted, label='Predicted')
plt.legend()
plt.show()

# Calculate errors
mse = mean_squared_error(test, predicted)
print("MSE = {:.4f}".format(mse))