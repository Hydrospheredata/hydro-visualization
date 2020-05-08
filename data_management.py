import json
import random
import tempfile
from time import sleep
from typing import Dict, Optional, List, Tuple

import boto3
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import s3fs
from hydrosdk.model import Model
from hydrosdk.servable import Servable
from loguru import logger as logging
from pymongo import MongoClient

from conf import AWS_STORAGE_ENDPOINT, HS_CLUSTER_ADDRESS, HYDRO_VIS_BUCKET_NAME, EMBEDDING_FIELD
from ml_transformers.transformer import Transformer
from ml_transformers.utils import DEFAULT_PARAMETERS, Coloring, get_top_N_neighbours


def get_mongo_client(mongo_url, mongo_port, mongo_user, mongo_pass, mongo_auth_db):
    return MongoClient(host=mongo_url, port=mongo_port, maxPoolSize=200,
                       username=mongo_user, password=mongo_pass,
                       authSource=mongo_auth_db)


class S3Manager:
    def __init__(self):

        if AWS_STORAGE_ENDPOINT:

            self.fs = s3fs.S3FileSystem(
                anon=False,
                client_kwargs={
                    'endpoint_url': AWS_STORAGE_ENDPOINT
                },
                config_kwargs={'s3': {'addressing_style': 'path'}}
            )

            from botocore.client import Config
            conf = Config(s3={'addressing_style': 'path'})

            boto_client = boto3.client(
                's3',
                endpoint_url=AWS_STORAGE_ENDPOINT,
                config=conf
            )

        else:
            self.fs = s3fs.S3FileSystem()
            boto_client = boto3.client(
                's3')
        sleep(random.random())
        if not self.fs.exists(f's3://{HYDRO_VIS_BUCKET_NAME}'):
            logging.info(f'Creating {HYDRO_VIS_BUCKET_NAME} bucket')
            try:
                boto_client.create_bucket(Bucket=HYDRO_VIS_BUCKET_NAME)
            except Exception as e:
                if not self.fs.exists(f's3://{HYDRO_VIS_BUCKET_NAME}'):
                    logging.error(f'Couldn\'t create {HYDRO_VIS_BUCKET_NAME} bucket due to error: {e}')

    def read_parquet(self, bucket_name, filename) -> Optional[pd.DataFrame]:
        if not bucket_name or not filename:
            return None
        try:
            df = pq.ParquetDataset(f's3://{bucket_name}/{filename}', filesystem=self.fs).read_pandas().to_pandas()
        except Exception as e:
            logging.error(f'Couldn\'t read parquet s3://{bucket_name}/{filename}. Error: {e}')
            return None
        return df

    def write_parquet(self, df: pd.DataFrame, bucket_name, filename):
        try:
            df.to_parquet(f's3://{bucket_name}/{filename}')
        except Exception as e:
            logging.error(f'Couldn\'t write {filename} to {bucket_name} bucket. Error: {e}')

    def write_json(self, data: Dict, filepath: str):
        with self.fs.open(filepath, 'w') as f:
            f.write(json.dumps(data))

    def read_json(self, filepath: str) -> Dict:
        '''
        Reads json
        If file not found or couldn't parse content, return empty Dict and log error
        :param bucket_name:
        :param filename:
        :return: file content or {} in case of error
        '''
        if not self.fs.exists(filepath):
            logging.error(f'No such file {filepath}')
            return {}
        clean_path = filepath.replace('s3://', '')
        with self.fs.open(clean_path, 'rb') as f:
            content = f.read()
        try:
            js_object = json.loads(content)
        except Exception as e:
            logging.error(f'Couldn\'t read {filepath} file as json file. Error: {e}')
            return {}
        return js_object

    def write_transformer_model(self, transformer, filepath) -> bool:
        """

        :param transformer:
        :param bucket_name:
        :param filename:
        :return:
        """
        with tempfile.TemporaryFile() as fp:
            try:
                joblib.dump(transformer, fp)
            except Exception as e:
                logging.error(f'Couldn\'t dump joblib file: {filepath}. Error: {e}')
                return False
            fp.seek(0)
            try:
                with self.fs.open(filepath, 'wb') as s3file:
                    s3file.write(fp.read())
            except Exception as e:
                logging.error(f'Couldn\'t write joblib file: {filepath}. Error: {e}')
                return False
        return True

    def read_transformer_model(self, bucket_name, filename) -> Optional[Transformer]:
        """
        Loads pretrained transformer model from S3
        if file not found: return None
        if is not joblib file: return None
        if not Transformer instance: return None
        :param bucket_name:
        :param filename:
        :return: Transformer/None
        """
        if not self.fs.exists(f's3://{bucket_name}/{filename}'):
            logging.error(f'No such file {bucket_name}/{filename}')
            return None
        with self.fs.open(f'{bucket_name}/{filename}', 'rb') as f:
            try:
                transformer = joblib.load(f)
            except Exception as e:
                logging.error(f'Couldn\'t load joblib model: {bucket_name}/{filename}')
                return None
            if not isinstance(transformer, Transformer):
                logging.error(f'{bucket_name}/{filename} ({transformer.__class__}) is not Transformer instance. ')
                return None
            return transformer


def parse_requests_dataframe(df, monitoring_fields: List[Tuple[str, str]], embeddings: np.ndarray) -> Dict:
    """
    Extracts:
        - class prediction,
        - confidence,
        - other monitoring metrics and thresholds
    from requests dataframe
    :param df: dataframe
    :param monitoring_fields: list of monitoring metrics names with comparison operator
    :return: Dict {"class_labels":{…}, "metrics":{…}}
    """

    def get_coloring_info(column: pd.Series) -> Dict:
        coloring_info = {}
        if np.issubdtype(column, np.integer):
            coloring = Coloring.CLASS
            coloring_info['classes'] = np.unique(column).tolist()
        elif np.issubdtype(column, np.floating):
            coloring = Coloring.GRADIENT
        else:
            coloring = Coloring.NONE
        coloring_info['coloring_type'] = coloring.value
        return coloring_info

    requests_ids = df['_id'].values.tolist()

    top_N_neighbours = get_top_N_neighbours(embeddings, N=50)
    counterfactuals = [[] for _ in range(len(top_N_neighbours))]
    predictions, confidence = [], []
    if 'class' in df.columns:
        predictions = {'data': df['class'].values.tolist()}
        predictions.update(get_coloring_info(df['class']))
        class_labels = df['class'].values.tolist()
        counterfactuals = list(
            map(lambda i: list(filter(lambda x: class_labels[x] != class_labels[i], top_N_neighbours[i])),
                range(len(top_N_neighbours))))
        del class_labels

    if 'confidence' in df.columns:
        confidence = {'data': df['confidence'].values.tolist()}
        confidence.update(get_coloring_info(df['confidence']))

    class_info = {'class': predictions, 'confidence': confidence}

    monitoring_data = {}
    metric_checks = df._hs_metric_checks.to_list()
    for (monitoring_metric_name, comparison_operator, threshold) in monitoring_fields:
        monitoring_data[monitoring_metric_name] = {'scores': [], 'threshold': threshold,
                                                   'operation': comparison_operator}

    for request in metric_checks:
        for (monitoring_metric_name, comparison_operator, threshold) in monitoring_fields:
            metric_dict = request.get(monitoring_metric_name, {})
            metric_data = monitoring_data[monitoring_metric_name]
            metric_data['scores'].append(metric_dict.get('value', None))
            monitoring_data[monitoring_metric_name] = metric_data

    for (monitoring_metric_name, comparison_operator, threshold) in monitoring_fields:
        monitoring_data[monitoring_metric_name].update(
            get_coloring_info(pd.Series(monitoring_data[monitoring_metric_name]['scores'])))

    return {'class_labels': class_info,
            'metrics': monitoring_data,
            'requests_ids': requests_ids,
            'top_N': top_N_neighbours,
            'counterfactuals': counterfactuals}


def parse_embeddings_from_dataframe(df):
    embeddings = np.apply_along_axis(lambda x: np.array(x), arr=df[EMBEDDING_FIELD].values.tolist(), axis=0)
    return embeddings


def get_record(db, method, model_name, model_version) -> Dict:
    existing_record = db[method].find_one({"model_name": model_name,
                                           "model_version": model_version})
    if not existing_record:
        return {"model_name": model_name,
                "model_version": model_version,
                "result_file": "",
                "transformer_file": "",
                "parameters": DEFAULT_PARAMETERS[method],
                "use_labels": False}
    else:
        return existing_record


def update_record(db, method, record, model_name, model_version):
    model_version = str(model_version)
    if '_id' in record:
        del record['_id']
    db[method].update_one({"model_name": model_name,
                           "model_version": model_version},
                          {"$set": record}, upsert=True)


def compute_training_embeddings(model: Model, servable: Servable, training_data: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Computes embeddings from training data using unmonitorable servable
    :param model: model instance
    :param servable: servable
    :param training_data: model training data dataframe
    :return: np.array [N, embedding_dim]
    """
    predictor = servable.predictor(monitorable=False)
    embeddings = []
    model_inputs_names = [input.name for input in model.contract.predict.inputs]
    n_samples = len(training_data)
    for i in range(n_samples):
        try:
            sample = training_data.iloc[i].to_dict()
            request = {k: v for k, v in sample.items() if k in model_inputs_names}
            result = predictor.predict(request)
        except Exception as e:
            logging.error(f'Couldn\'t get prediction of data sample: {e}')
            return None
        embeddings.append(result[EMBEDDING_FIELD])
    embeddings = np.concatenate(embeddings, axis=0)
    logging.info(f'Inferenced training data. Embeddings shape:  {embeddings.shape}')
    return embeddings


def get_production_subsample(model_id, size=1000) -> pd.DataFrame:
    r = requests.get(f'{HS_CLUSTER_ADDRESS}/monitoring/checks/subsample/{model_id}?size={size}')
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(r.json())
