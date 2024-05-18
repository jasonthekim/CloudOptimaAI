import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, BatchNormalization, Dropout, Bidirectional
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

metrics_data = pd.read_csv('.\code\(not)realistic_mock_cloudwatch_data_90days.csv')
cost_data = pd.read_csv('.\code\(not)realistic_mock_cost_data_90days.csv')

metrics_data['Date'] = pd.to_datetime(metrics_data['Timestamp']).dt.date
cost_data['Date'] = pd.to_datetime(cost_data['Timestamp']).dt.date
metrics_data.drop('Timestamp', axis=1, inplace=True)
cost_data.drop('Timestamp', axis=1, inplace=True)

daily_metrics = metrics_data.groupby('Date').mean().reset_index() 
daily_cost = cost_data.groupby('Date')['Cost'].sum().reset_index()
daily_data = pd.merge(daily_metrics, daily_cost, on='Date', how='outer')
numeric_cols = daily_metrics.select_dtypes(include=[np.number]).columns

scaler = StandardScaler()
daily_data[numeric_cols] = scaler.fit_transform(daily_data[numeric_cols])


N = 60  # past days used to predict

X, y = [], []
for i in range(N, len(daily_data)):
    X.append(daily_data.iloc[i - N:i][numeric_cols].values)
    y.append(daily_data.iloc[i]['Cost'])

assert len(X) == len(y), "The features and target should have the same number of samples"

print(len(X), len(y) , "len of X and y")

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# convert to 3D array 
X_train = np.array(X_train)
X_test = np.array(X_test)

print(f"Shape of X_train: {X_train.shape}")  # should be (number_of_samples, N, number_of_features)
print(f"Shape of X_test: {X_test.shape}") 

model = Sequential()

model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=(N, len(numeric_cols)), kernel_regularizer=l2(0.001), recurrent_dropout=0.1))
model.add(BatchNormalization())  
model.add(Dropout(0.1))
model.add(Bidirectional(LSTM(64, activation='tanh', kernel_regularizer=l2(0.001))))
model.add(BatchNormalization()) 
model.add(Dropout(0.1))
model.add(Dense(1))

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

opt = Adam(learning_rate=0.001)
model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='mse')  

history = model.fit(
    X_train, y_train, 
    epochs=200, 
    verbose=1, 
    batch_size=32,  
    validation_data=(X_val, y_val), 
    callbacks=[early_stopping, reduce_lr]
)

#eval
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

model.save('code/cloud_cost_prediction_model.h5')
joblib.dump(scaler, 'code/scaler.save')


"""
visualizations:
"""


plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


plt.figure(figsize=(10,6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Cost Prediction')
plt.xlabel('Samples')
plt.ylabel('Cost')
plt.legend()
plt.show()


errors = y_test - y_pred.flatten()
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True)
plt.title('Prediction Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.show()



plt.figure(figsize=(10, 8))
correlation_matrix = daily_data[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


########################################################################################################################

