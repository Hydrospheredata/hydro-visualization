from boto3.session import Session
from loguru import logger
import boto3
import os
from io import StringIO
import pandas as pd
import numpy as np


class S3Manager:
    def __init__(self):
        self.access_key = os.getenv('AWS_ACCESS_KEY', '')
        self.secret_key = os.getenv('AWS_SECRET_KEY', '')
        if self.access_key == '' or self.secret_key == '':
            logger.info('Access keys for S3 were not found')
        self.s3 = boto3.client('s3', aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_key)
        self.s3 = boto3.client('s3')

    def read_file(self, bucket_name, filename):
        response = self.s3.get_object(Bucket=bucket_name,
                                      Key=filename)
        object_body = response['Body']
        file_type = response['ContentType']
        if file_type == 'text/csv':
            csv_string = object_body.read().decode('utf-8')
            if len(csv_string) == 0:
                return '', None
            df = pd.read_csv(StringIO(csv_string))
            return file_type, df
        return '', None

    def upload(self, bucket, file, path):
        pass


def parse_requests_dataframe(df):
    '''
    TODO change extracted fields
    Extracts:
        - embeddings,
        - class prediction,
        - confidence,
        - anomaly class, # TODO
        - anomaly confidence # TODO
    from requests dataframe
    :param df: dataframe
    :return: np.arrays
    '''
    predictions = np.array(df['prediction'])
    confidence = np.array(df['prob'])
    embeddings = np.array([np.fromstring(df['embs'][i][1:-1], dtype=np.float32, sep=',') for i in range(len(df['embs']))])
    return embeddings, predictions, confidence, [], []

def get_requests_data(bucket_name, requests_files):
    s3manager = S3Manager()
    embeddings = []
    predictions = []
    confidences = []
    anomaly_predictions = []
    anomaly_confidences = []
    for file in requests_files:
        print(file, bucket_name)
        _, requests_df = s3manager.read_file(bucket_name=bucket_name, filename=file)
        if requests_df is None:
            logger.error(f'Could not get {bucket_name}:/{file}')
            continue
        r_embeddings, r_predictions, r_confidences, r_anomaly_preds, r_anomaly_conf = parse_requests_dataframe(requests_df)
        embeddings.append(r_embeddings)
        predictions.append(r_predictions)
        confidences.append(r_confidences)
        anomaly_predictions.append(r_anomaly_preds)
        anomaly_confidences.append(r_anomaly_conf)
    embeddings = np.concatenate(embeddings)
    predictions = np.concatenate(predictions)
    confidences = np.concatenate(confidences)
    anomaly_predictions = np.concatenate(anomaly_predictions)
    anomaly_confidences = np.concatenate(anomaly_confidences)
    return embeddings, predictions, confidences, anomaly_predictions, anomaly_confidences
