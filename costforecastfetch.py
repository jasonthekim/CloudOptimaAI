import boto3
import csv
from datetime import datetime, timedelta
import pandas as pd

def fetch_cloudwatch_data(resources, start_time, end_time):
    cloudwatch = boto3.client('cloudwatch')

    metrics_info = {
        'EC2': [
            {'metric_name': 'CPUUtilization', 'stat': 'Average', 'dimensions': [{'Name': 'InstanceId', 'Value': resources['EC2']['instance_id']}]},
            {'metric_name': 'NetworkPacketsOut', 'stat': 'Sum', 'dimensions': [{'Name': 'InstanceId', 'Value': resources['EC2']['instance_id']}]},
            {'metric_name': 'DiskWriteOps', 'stat': 'Sum', 'dimensions': [{'Name': 'InstanceId', 'Value': resources['EC2']['instance_id']}]},
            {'metric_name': 'NetworkBytesIn', 'stat': 'Average', 'dimensions': [{'Name': 'InstanceId', 'Value': resources['EC2']['instance_id']}]},
        ],
        'S3': [
            {'metric_name': 'BucketSizeBytes', 'stat': 'Average', 'dimensions': [{'Name': 'BucketName', 'Value': resources['S3']['bucket_name']}, {'Name': 'StorageType', 'Value': 'StandardStorage'}]},
            {'metric_name': 'NumberOfObjects', 'stat': 'Average', 'dimensions': [{'Name': 'BucketName', 'Value': resources['S3']['bucket_name']}, {'Name': 'StorageType', 'Value': 'AllStorageTypes'}]},
            {'metric_name': '4xxErrors', 'stat': 'Sum', 'dimensions': [{'Name': 'BucketName', 'Value': resources['S3']['bucket_name']}]},
            {'metric_name': '5xxErrors', 'stat': 'Sum', 'dimensions': [{'Name': 'BucketName', 'Value': resources['S3']['bucket_name']}]},        
        ],
        'Redshift': [
            {'metric_name': 'CPUUtilization', 'stat': 'Average', 'dimensions': [{'Name': 'ClusterIdentifier', 'Value': resources['Redshift']['cluster_id']}]},
            {'metric_name': 'DatabaseConnections', 'stat': 'Average', 'dimensions': [{'Name': 'ClusterIdentifier', 'Value': resources['Redshift']['cluster_id']}]},
            {'metric_name': 'NetworkReceiveThroughput', 'stat': 'Average', 'dimensions': [{'Name': 'ClusterIdentifier', 'Value': resources['Redshift']['cluster_id']}]},
        ],
        'Kinesis': [
            {'metric_name': 'GetRecords.IteratorAgeMilliseconds', 'stat': 'Average', 'dimensions': [{'Name': 'StreamName', 'Value': resources['Kinesis']['stream_name']}]},
            {'metric_name': 'PutRecord.Success', 'stat': 'Average', 'dimensions': [{'Name': 'StreamName', 'Value': resources['Kinesis']['stream_name']}]},
            {'metric_name': 'GetRecords.Bytes', 'stat': 'Average', 'dimensions': [{'Name': 'StreamName', 'Value': resources['Kinesis']['stream_name']}]},        
        ],
        'Lambda': [
            {'metric_name': 'Duration', 'stat': 'Average', 'dimensions': [{'Name': 'FunctionName', 'Value': resources['Lambda']['function_name']}]},
            {'metric_name': 'Errors', 'stat': 'Sum', 'dimensions': [{'Name': 'FunctionName', 'Value': resources['Lambda']['function_name']}]},
            {'metric_name': 'ConcurrentExecutions', 'stat': 'Average', 'dimensions': [{'Name': 'FunctionName', 'Value': resources['Lambda']['function_name']}]},
            {'metric_name': 'DeadLetterErrors', 'stat': 'Sum', 'dimensions': [{'Name': 'FunctionName', 'Value': resources['Lambda']['function_name']}]},
        ],
        'DynamoDB': [
            {'metric_name': 'ReadThrottleEvents', 'stat': 'Sum', 'dimensions': [{'Name': 'TableName', 'Value': resources['DynamoDB']['table_name']}]},
            {'metric_name': 'WriteThrottleEvents', 'stat': 'Sum', 'dimensions': [{'Name': 'TableName', 'Value': resources['DynamoDB']['table_name']}]},
            {'metric_name': 'ProvisionedReadCapacityUnits', 'stat': 'Average', 'dimensions': [{'Name': 'TableName', 'Value': resources['DynamoDB']['table_name']}]},
            {'metric_name': 'ProvisionedWriteCapacityUnits', 'stat': 'Average', 'dimensions': [{'Name': 'TableName', 'Value': resources['DynamoDB']['table_name']}]},
        ]
    }

    metric_data_queries = []

    for service, metrics in metrics_info.items():
        for metric in metrics:
            query = {
                'Id': f"{service}_{metric['metric_name']}",
                'MetricStat': {
                    'Metric': {
                        'Namespace': f"AWS/{service}",
                        'MetricName': metric['metric_name'],
                        'Dimensions': metric['dimensions']
                    },
                    'Period': 300,  # 5 minutes in seconds
                    'Stat': metric['stat']
                },
                'ReturnData': True,
            }
            metric_data_queries.append(query)

    response = cloudwatch.get_metric_data(
        MetricDataQueries=metric_data_queries,
        StartTime=start_time,
        EndTime=end_time
    )

    with open('cloudwatch_data.csv', 'w', newline='') as csvfile:
        fieldnames = ['Timestamp'] + [f"{service}_{metric['metric_name']}" for service, metrics in metrics_info.items() for metric in metrics]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, timestamp in enumerate(response['MetricDataResults'][0]['Timestamps']):
            row_data = {'Timestamp': timestamp}
            for result in response['MetricDataResults']:
                row_data[result['Id']] = result['Values'][i]
            writer.writerow(row_data)

    print("Data written to cloudwatch_data.csv")

def get_cost_data(start_date, end_date):
    session = boto3.Session()
    client = session.client('ce')
    
    # Fetch daily cost
    response = client.get_cost_and_usage(
        TimePeriod={
            'Start': start_date,
            'End': end_date
        },
        Granularity='DAILY',
        Metrics=['UnblendedCost'],
        GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
    )
    
    # Process the response
    dates = []
    costs = []
    
    for result in response['ResultsByTime']:
        for group in result['Groups']:
            dates.append(result['TimePeriod']['Start'])
            costs.append(float(group['Metrics']['UnblendedCost']['Amount']))
    
    df = pd.DataFrame({'Date': dates, 'Cost': costs})
    return df


def dispatchfetch(start_date, end_date):
    RESOURCES = {
        'EC2': {'instance_id': 'YOUR_INSTANCE_ID'},
        'S3': {'bucket_name': 'YOUR_S3_BUCKET_NAME'},
        'Redshift': {'cluster_id': 'YOUR_REDSHIFT_CLUSTER_ID'},
        'Kinesis': {'stream_name': 'YOUR_KINESIS_STREAM_NAME'},
        'Lambda': {'function_name': 'YOUR_LAMBDA_FUNCTION_NAME'},
        'DynamoDB': {'table_name': 'YOUR_DYNAMODB_TABLE_NAME'}
    }
    
    START_TIME = datetime.now() - timedelta(days=7)  # 1 week ago
    END_TIME = datetime.now()
    
    fetch_cloudwatch_data(RESOURCES, START_TIME, END_TIME)
    get_cost_data(START_TIME,END_TIME)
    
    cost_data = get_cost_data(start_date, end_date)
    cost_data.to_csv('cost_data.csv', index=False)
    print("Cost data has been saved to 'cost_data.csv'.")


