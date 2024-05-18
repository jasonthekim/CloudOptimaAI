import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import joblib
import matplotlib.pyplot as plt

scaler = joblib.load('code/scaler.save')
inputdatacloudwatch = pd.read_csv('code\(not)realistic_mock_cloudwatch_data_90days.csv')
inputdatacost = pd.read_csv('code\(not)realistic_mock_cost_data_90days.csv')
inputdatacloudwatch['Date'] = pd.to_datetime(inputdatacloudwatch['Timestamp']).dt.date
inputdatacloudwatch.drop('Timestamp', axis=1, inplace=True)
inputdatacost['Date'] = pd.to_datetime(inputdatacost['Timestamp']).dt.date
inputdatacost.drop('Timestamp', axis=1, inplace=True)

daily_metrics_input = inputdatacloudwatch.groupby('Date').mean().reset_index()
daily_cost_input = inputdatacost.groupby('Date')['Cost'].sum().reset_index()
daily_data_input = pd.merge(daily_metrics_input, daily_cost_input, on='Date', how='outer')
print("Size of daily_metrics_input:", daily_metrics_input.shape)
print("Size of daily_cost_input:", daily_cost_input.shape)
print("Size of daily_data_input before scaling:", daily_data_input.shape)

numeric_cols = daily_metrics_input.select_dtypes(include=[np.number]).columns
daily_data_input[numeric_cols] = scaler.transform(daily_data_input[numeric_cols])
print("Size of daily_data_input after scaling:", daily_data_input.shape)

N = 60  
input_X = []
prediction_days = 350  

for i in range(len(daily_data_input) - prediction_days, len(daily_data_input)):
    input_X.append(daily_data_input.iloc[i - N:i][numeric_cols].values)

input_X = np.array(input_X)
print("Size of input_X:", input_X.shape)
print("Any NaNs in input_X:", np.isnan(input_X).any())

model = load_model('code\cloud_cost_prediction_model.h5')
predictions = model.predict(input_X)
print(predictions)

plt.figure(figsize=(10, 6)) 
plt.plot(predictions, color='blue', label='Predicted Costs')  
plt.title('Predicted Cost Over Time for Next Year')  
plt.xlabel('Time (in days)')  
plt.ylabel('Cost')  
plt.legend()  
plt.show()  
