import random
import boto3
from botocore.exceptions import ClientError
from dataclasses import dataclass
from keras.models import load_model
import numpy as np

def fetch_ec2_config():
    ec2 = boto3.client('ec2')
    try:
        instances = ec2.describe_instances()
        return instances
    except ClientError as e:
        print(f"Error fetching EC2 config: {e}")
        return None

def fetch_s3_config():
    s3 = boto3.client('s3')
    try:
        buckets = s3.list_buckets()
        return buckets
    except ClientError as e:
        print(f"Error fetching S3 config: {e}")
        return None

def fetch_redshift_config():
    redshift = boto3.client('redshift')
    try:
        clusters = redshift.describe_clusters()
        return clusters
    except ClientError as e:
        print(f"Error fetching Redshift config: {e}")
        return None

def fetch_kinesis_config():
    kinesis = boto3.client('kinesis')
    try:
        streams = kinesis.list_streams()
        return streams
    except ClientError as e:
        print(f"Error fetching Kinesis config: {e}")
        return None

def fetch_lambda_config():
    lambda_client = boto3.client('lambda')
    try:
        functions = lambda_client.list_functions()
        return functions
    except ClientError as e:
        print(f"Error fetching Lambda config: {e}")
        return None

def fetch_dynamodb_config():
    dynamodb = boto3.client('dynamodb')
    try:
        tables = dynamodb.list_tables()
        return tables
    except ClientError as e:
        print(f"Error fetching DynamoDB config: {e}")
        return None

@dataclass
class AWSConfiguration:
    ec2_config: dict
    s3_config: dict
    redshift_config: dict
    kinesis_config: dict
    lambda_config: dict
    dynamodb_config: dict

def generate_initial_population(size):
    population = []
    for _ in range(size):
        config = AWSConfiguration(
            ec2_config=fetch_ec2_config(),
            s3_config=fetch_s3_config(),
            redshift_config=fetch_redshift_config(),
            kinesis_config=fetch_kinesis_config(),
            lambda_config=fetch_lambda_config(),
            dynamodb_config=fetch_dynamodb_config()
        )
        population.append(config)
    return population

model = load_model('code\optimalcostmodel.h5')

def calculate_fitness(configuration):
    predicted_fitness = model.predict(np.array([configuration]))
    return predicted_fitness

def select(population, top_n):
    top_n = min(top_n, len(population))
    sorted_population = sorted(population, key=calculate_fitness, reverse=True)
    return sorted_population[:top_n]

def crossover(parent1, parent2):
    child_config = {}
    for key in parent1:
        if random.random() > 0.5:
            child_config[key] = parent1[key]
        else:
            child_config[key] = parent2[key]
    return child_config

def mutate(config):
    mutation_probability = 0.1 
    for key in config:
        if random.random() < mutation_probability:
            change_factor = 1 + random.uniform(-0.1, 0.1)  
            config[key] *= change_factor
    return config

def genetic_algorithm(population_size, generations):
    population = generate_initial_population(population_size)
    for _ in range(generations):
        selected = select(population)
        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_generation.append(child)
        population = next_generation
    return max(population, key=calculate_fitness)

optimal_config = genetic_algorithm(population_size=100, generations=50)


