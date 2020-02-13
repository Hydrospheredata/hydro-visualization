import json
import os
import pickle
import tempfile
from datetime import datetime
from io import StringIO
from typing import Dict, Optional

import boto3
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import s3fs
from bson.binary import Binary
from loguru import logger as logging
from tqdm import tqdm

from client import HydroServingModel, HydroServingServable
from ml_transformers.transformer import Transformer
from ml_transformers.utils import DEFAULT_PARAMETERS


class S3Manager:
    def __init__(self):
        self.access_key = os.getenv('AWS_ACCESS_KEY', '')
        self.secret_key = os.getenv('AWS_SECRET_KEY', '')
        if self.access_key == '' or self.secret_key == '':
            logging.info('Access keys for S3 were not found')
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

    def write_parquet(self, df: pd.DataFrame, bucket_name, filename):
        try:
            df.to_parquet(f's3://{bucket_name}/{filename}')
        except Exception as e:
            logging.error(f'Couldn\'t write {filename} to {bucket_name} bucket. Error: {e}')

    def write_json(self, data: Dict, bucket_name: str, filename: str) -> bool:
        try:
            self.s3.put_object(Bucket=bucket_name, Body=json.dumps(data), Key=filename)
        except Exception as e:
            logging.error(f'Couldn\'t write json to {bucket_name} bucket. Error: {e}')
            return False
        return True

    def read_json(self, bucket_name: str, filename: str) -> Dict:
        try:
            response = self.s3.get_object(Bucket=bucket_name, Key=filename)
        except Exception as e:
            logging.error(f'Couldn\'t request {bucket_name} bucket. Error: {e}')
            return {}
        object_body = response['Body']
        try:
            content = json.loads(object_body.read())
        except Exception as e:
            logging.error(f'Couldn\'t read {bucket_name}:{filename} file as json file. Error: {e}')
            return {}
        return content

    def file_exists(self, bucket_name: str, filename):
        try:
            response = self.s3.list_objects(Bucket=bucket_name)
        except Exception as e:
            logging.error(f'Couldn\'t request {bucket_name} bucket. Error: {e}')
            return False
        files = list(map(lambda x: x.get('Key', ''), response.get('Contents', [{}])))
        if filename in files:
            return True
        return False

    def write_transformer(self, transformer, bucket_name, filename):
        try:
            with tempfile.TemporaryFile() as fp:
                joblib.dump(transformer, fp)
                fp.seek(0)
                self.s3.put_object(Body=fp.read(), Bucket=bucket_name, Key=filename)
        except Exception as e:
            logging.error(f'Couldn\'t write transformer to {bucket_name}/{filename} bucket. Error: {e}')
            return False
        return True

    def read_transformer(self, bucket_name, filename) -> Optional[Transformer]:
        '''
        Loads pretrained transformer model from S3
        :param bucket_name:
        :param filename:
        :return: transofrmer or None (if file doesn't exist)
        '''
        if not self.file_exists(bucket_name, filename):
            logging.error('Transformer doesn\'t exists')
            return None
        with tempfile.TemporaryFile() as fp:
            self.s3.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=filename)
            fp.seek(0)
            transformer = joblib.load(fp)
            return transformer


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
                               "operation": "Grater"
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
    for (monitoring_name, comp_operator) in monitoring_fields:
        scores = np.array(df[monitoring_name])
        thresholds = np.array(df[f'{monitoring_name}_threshold'])
        monitoring[monitoring_name] = {'scores': scores.tolist(), 'threshold': thresholds[0],
                                       'operation': comp_operator}  # taking last threshold
    class_info = {'class': predictions, 'confidence': confidence}
    return {'class_labels': class_info, 'metrics': monitoring, 'requests_ids': requests_ids}, embeddings


def get_requests_data(requests_df: pd.DataFrame, monitoring_fields) -> Dict:
    '''

    :param bucket_name:
    :param requests_files:
    :param monitoring_fields:
    :return: Dict
    '''

    if 'embedding' not in requests_df.columns:
        logging.error(f'No "embedding" field')
        return None

    return parse_requests_dataframe(requests_df, monitoring_fields)


def get_pretrained_files(db, method, model_name, model_version):
    existing_record = db[method].find_one({"model_name": model_name,
                                           "model_version": model_version})

    if existing_record is not None:
        model_record = existing_record.get('model', {})
        model_object = deserialize(model_record['object'])
        return model_object
    return None


def get_record(db, method, model_name, model_version):
    existing_record = db[method].find_one({"model_name": model_name,
                                           "model_version": model_version})
    if existing_record is None:
        return {"model_name": model_name,
                "model_version": model_version,
                "embeddings_bucket_name": "",
                "result_file": "",
                "transformer_file": "",
                "parameters": DEFAULT_PARAMETERS[method],
                "use_labels": False}
    return existing_record


def save_record(db, method: str, record: Dict):
    '''
    Saves information about transformer to database or updates information if it existed
    :param db:
    :param method: 'umap' (method name from ml_transformers.utils.AVAILABLE_TRANSFORMERS)
    :param record: Dict with values
    :return:
    '''
    model_name = record['model_name']
    model_version = record['model_version']
    existing_record = db[method].find_one({"model_name": model_name,
                                           "model_version": model_version})
    if existing_record:
        logging.info('Record existed!')
        db[method].update_one({"model_name": model_name,
                               "model_version": model_version}, {"$set": record})
    else:
        logging.info('Saving new record')
        db[method].insert_one(record)


def save_instance(db, method, model_name, model_version, instance) -> bool:
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
    logging.info(f'saved serialized model')
    return True


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


def get_training_embeddings(model: HydroServingModel, servable: HydroServingServable,
                            training_data: pd.DataFrame) -> [np.ndarray, None]:
    '''
    Computes embeddings from training data using model servable
    :param model:
    :param servable:
    :param training_data: Dataframe with data
    :return: np.ndarray with embeddings or None if failed
    '''
    if 'embedding' in training_data.columns:
        logging.info('Training embeddings exist')
        embs = np.stack(training_data['embedding'].values)
        return embs
    output_names = list(map(lambda x: x['name'], model.contract.contract_dict['outputs']))
    input_names = list(map(lambda x: x['name'], model.contract.contract_dict['inputs']))
    if 'embedding' not in output_names:
        raise ValueError(f'No output called "embedding" in model {model}')

    if len(set(input_names) - set(training_data.columns)) != 0:
        raise ValueError(
            f'Input fields ({input_names}) are not compatible with data fields ({set(training_data.columns)})')
    inputs = training_data[input_names]
    embeddings = []
    logging.info('Embedding inference: ')
    for i in tqdm(range(len(inputs))):
        outputs = servable(inputs.iloc[i])
        embeddings.append(outputs['embedding'])

    return np.concatenate(embeddings)
