import os
import pickle
from datetime import datetime
from io import StringIO
from typing import Dict

import boto3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import s3fs
from bson.binary import Binary
from loguru import logger
from tqdm import tqdm

from client import HydroServingClient, HydroServingModel, HydroServingServable


class S3Manager:
    def __init__(self):
        self.access_key = os.getenv('AWS_ACCESS_KEY', '')
        self.secret_key = os.getenv('AWS_SECRET_KEY', '')
        if self.access_key == '' or self.secret_key == '':
            logger.info('Access keys for S3 were not found')
        self.s3 = boto3.client('s3', aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_key)
        self.s3 = boto3.client('s3')
        self.fs = s3fs.S3FileSystem()

    def read_parquet(self, bucket_name, filename):
        return pq.ParquetDataset(f's3://{bucket_name}/{filename}', filesystem=self.fs).read_pandas().to_pandas()

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



def parse_requests_dataframe(df, monitoring_fields) -> (Dict, np.ndarray):
    '''
    Extracts:
        - embeddings,
        - class prediction,
        - confidence,
        - other monitoring metrics and thresholds
    from requests dataframe
    :param df: dataframe
    :return: Dict
    class_labels": {
                     "confidences": [0.1, 0.2, 0.3],
                     "predicted": [1, 2, 1, 2],
                     "ground_truth": [1, 1, 1, 2]
                       },
     "metrics": {
                 "anomality": {
                               "scores": [0.1, 0.2, 0.5, 0.2],
                               "threshold": 0.5
                               }
                 }
    }
    Embedding np.ndarray
    '''
    monitoring = {}
    predictions, confidence = [], []
    if 'class' in df.columns:
        predictions = df['class'].values.tolist()
    if 'confidence' in df.columns:
        confidence = df['confidence'].values.tolist()
    requests_ids = df['request_id'].values.tolist()
    embeddings = np.apply_along_axis(lambda x: np.array(x), arr=df['embedding'].values.tolist(), axis=0)
    for monitoring_name in monitoring_fields:
        scores = np.array(df[monitoring_name])
        thresholds = np.array(df[f'{monitoring_name}_threshold'])
        monitoring[monitoring_name] = {'scores': scores.tolist(), 'threshold': thresholds[0]}  # taking last threshold
    class_info = {'class': predictions, 'confidence': confidence}
    return {'class_labels': class_info, 'metrics': monitoring, 'requests_ids': requests_ids}, embeddings


def get_requests_data(requests_df, monitoring_fields) -> Dict:
    '''

    :param bucket_name:
    :param requests_files:
    :param monitoring_fields:
    :return: Dict
    '''

    if 'embedding' not in requests_df.columns:
        logger.error(f'No "embedding" field')
        return None

    return parse_requests_dataframe(requests_df, monitoring_fields)


def model_exists(db, method, model_name, model_version):
    existing_record = db[method].find_one({"model_name": model_name,
                                           "model_version": model_version})
    if existing_record is not None:
        model_record = existing_record.get('model', {})
        if model_record:
            return True
    return False


def get_pretrained_files(db, method, model_name, model_version):
    existing_record = db[method].find_one({"model_name": model_name,
                                           "model_version": model_version})

    if existing_record is not None:
        model_record = existing_record.get('model', {})
        model_object = deserialize(model_record['object'])
        transformed_emb_files = model_record['transformed_embeddings']  # TODO
        # if transformed_emb_files:
        return model_object
    return None


def get_record(db, method, model_name, model_version):
    existing_record = db[method].find_one({"model_name": model_name,
                                           "model_version": model_version})
    if existing_record is None:
        return {}
    return existing_record


def save_instance(db, method, model_name, model_version, instance):
    existing_record = db[method].find_one({"model_name": model_name,
                                           "model_version": model_version})
    if existing_record is None:
        return False
    model_record = existing_record.get('model', {})
    model_record['created'] = datetime.now()
    model_record['object'] = serialize(instance)
    existing_record['model'] = model_record
    db[method].update_one({"model_name": model_name,
                           "model_version": model_version}, {"$set": existing_record})
    logger.info(f'saved serialized model')
    return True


def get_record_status(db, method, model_name, model_version):
    '''
    returns info about:
     - if db already has info about model
     - if there is pretrained model in db
    :param db:
    :param method:
    :param model_name:
    :param model_version:
    :return: {record: True, model: True}
    '''
    status = {}
    existing_record = db[method].find_one({"model_name": model_name,
                                           "model_version": model_version})
    if existing_record is not None:
        status['record'] = True
        model_record = existing_record.get('model', {})
        if model_record:
            status['model'] = True
        else:
            status['model'] = False
    else:
        status['record'] = False
        status['model'] = False


def serialize(obj):
    '''
    Serializes object to write it to database
    :param obj: custom object
    :return: bytes string
    '''
    return Binary(pickle.dumps(obj))


def deserialize(bytes_str):
    '''
    Deserializes object written to db to obj
    :param bytes_str:
    :return: object
    '''
    return pickle.loads(bytes_str)


def get_model_monitoring_fields(model_name, model_version, hs_client: HydroServingClient):
    try:
        model = hs_client.get_model(model_name, model_version)
    except ValueError:  # no such model
        print('Value Error')
        return []
    model_id = model.id
    request_url = hs_client.url.split(':')[0]
    monitoring_response = requests.get(f'https://{request_url}/api/v2/monitoring/metricspec/modelversion/{model_id}')
    if monitoring_response.status_code == 200:
        return list(map(lambda x: x['name'], monitoring_response.json()))
    else:

        return []


def get_training_embeddings(model: HydroServingModel, servable: HydroServingServable,
                            training_data: pd.DataFrame) -> [np.ndarray, None]:
    output_names = list(map(lambda x: x['name'], model.contract.contract_dict['outputs']))
    input_names = list(map(lambda x: x['name'], model.contract.contract_dict['inputs']))
    if 'embedding' not in output_names:
        raise ValueError(f'No output called "embedding" in model {model}')

    if len(set(input_names) - set(training_data.columns)) != 0:
        raise ValueError(
            f'Input fields ({input_names}) are not compatible with data fields ({set(training_data.columns)})')
    inputs = training_data[input_names]
    embeddings = []
    logger.info('Embedding inference: ')
    for i in tqdm(range(len(inputs))):
        outputs = servable(inputs.iloc[i])
        embeddings.append(outputs['embedding'])

    return np.concatenate(embeddings)
