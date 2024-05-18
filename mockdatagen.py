import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# generate cloudwatch data
def generate_timeseries_data(length, trend_factor=0.1, seasonality_factor=5, noise_factor=0.5):
    t = np.arange(length)
    trend = t * trend_factor
    noise = np.random.randn(length) * noise_factor
    seasonality = np.sin(2 * np.pi * t / 24) * seasonality_factor 
    weekly_seasonality = np.sin(2 * np.pi * t / (24*7)) * seasonality_factor * 0.5

    return trend + seasonality + weekly_seasonality + noise



start_time = datetime.now() - timedelta(days=1080)
end_time = datetime.now()
time_range = pd.date_range(start_time, end_time, freq='5T')  # 5 minute frequency here as well

metrics_info = {
    'EC2': ['CPUUtilization', 'NetworkPacketsOut', 'DiskWriteOps', 'NetworkBytesIn'],
    'S3': ['BucketSizeBytes', 'NumberOfObjects', '4xxErrors', '5xxErrors'],
    'Redshift': ['CPUUtilization', 'DatabaseConnections', 'NetworkReceiveThroughput'],
    'Kinesis': ['GetRecords.IteratorAgeMilliseconds', 'PutRecord.Success', 'GetRecords.Bytes'],
    'Lambda': ['Duration', 'Errors', 'ConcurrentExecutions', 'DeadLetterErrors'],
    'DynamoDB': ['ReadThrottleEvents', 'WriteThrottleEvents', 'ProvisionedReadCapacityUnits', 'ProvisionedWriteCapacityUnits']
}

# generate metrics
data = {'Timestamp': time_range}
for service, metrics in metrics_info.items():
    for metric in metrics:
        data[f"{service}_{metric}"] = generate_timeseries_data(len(time_range))

df = pd.DataFrame(data)
df.to_csv('(not)realistic_mock_cloudwatch_data_90days.csv', index=False)
print("Realistic mock CloudWatch data saved to 'realistic_mock_cloudwatch_data_90days.csv'")






def generate_cost_data(length, base_cost=50, trend_factor=0.1, seasonality_factor=30, noise_factor=2):
    t = np.arange(length)

    trend = t * trend_factor
    seasonality = np.sin(2 * np.pi * t / 24) * seasonality_factor 
    weekly_seasonality = np.sin(2 * np.pi * t / (24*7)) * seasonality_factor * 0.5
    noise = np.random.randn(length) * noise_factor

    return base_cost + trend + seasonality + weekly_seasonality + noise

dates = pd.date_range(start_time, end_time, freq='D')  # daily frequency
costs = generate_cost_data(len(dates))

cost_df = pd.DataFrame({'Timestamp': dates, 'Cost': costs})
cost_df.to_csv('(not)realistic_mock_cost_data_90days.csv', index=False)
print("Realistic mock cost data saved to 'realistic_mock_cost_data_90days.csv'")
