import json
import tempfile
from typing import Dict, Optional, List, Tuple

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import s3fs
from hydrosdk.model import Model
from loguru import logger as logging
from pymongo import MongoClient
from tqdm import tqdm

from client import HydroServingServable
from ml_transformers.transformer import Transformer
from ml_transformers.utils import DEFAULT_PARAMETERS, Coloring, get_top_100_neighbours


def get_mongo_client(mongo_url, mongo_port, mongo_user, mongo_pass, mongo_auth_db):
    return MongoClient(host=mongo_url, port=mongo_port, maxPoolSize=200,
                       username=mongo_user, password=mongo_pass,
                       authSource=mongo_auth_db)


class S3Manager:
    def __init__(self):
        self.fs = s3fs.S3FileSystem()

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

    def write_json(self, data: Dict, bucket_name: str, filename: str):
        with self.fs.open(f's3://{bucket_name}/{filename}', 'w') as f:
            f.write(json.dumps(data))

    def read_json(self, bucket_name: str, filename: str) -> Dict:
        '''
        Reads json
        If file not found or couldn't parse content, return empty Dict and log error
        :param bucket_name:
        :param filename:
        :return: file content or {} in case of error
        '''
        if not self.fs.exists(f's3://{bucket_name}/{filename}'):
            logging.error(f'No such file {bucket_name}/{filename}')
            return {}
        with self.fs.open(f'{bucket_name}/{filename}', 'rb') as f:
            content = f.read()
        try:
            js_object = json.loads(content)
        except Exception as e:
            logging.error(f'Couldn\'t read {bucket_name}:{filename} file as json file. Error: {e}')
            return {}
        return js_object

    def write_transformer_model(self, transformer, bucket_name, filename) -> bool:
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
                logging.error(f'Couldn\'t dump joblib file: {bucket_name}/{filename}. Error: {e}')
                return False
            fp.seek(0)
            try:
                with self.fs.open(f's3://{bucket_name}/{filename}', 'wb') as s3file:
                    s3file.write(fp.read())
            except Exception as e:
                logging.error(f'Couldn\'t write joblib file: {bucket_name}/{filename}. Error: {e}')
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
            coloring_info['classes'] = np.unique(df['class']).tolist()
        elif np.issubdtype(column, np.floating):
            coloring = Coloring.GRADIENT
        else:
            coloring = Coloring.NONE
        coloring_info['coloring_type'] = coloring.value
        return coloring_info

    requests_ids = df['request_id'].values.tolist()

    top_100_neighbours = get_top_100_neighbours(embeddings)
    counterfactuals = [[] for _ in range(len(top_100_neighbours))]
    predictions, confidence = [], []
    if 'class' in df.columns:
        predictions = {'data': df['class'].values.tolist()}
        predictions.update(get_coloring_info(df['class']))
        class_labels = df['class'].values.tolist()
        counterfactuals = list(
            map(lambda i: list(filter(lambda x: class_labels[x] != class_labels[i], top_100_neighbours[i])),
                range(len(top_100_neighbours))))
        del class_labels

    if 'confidence' in df.columns:
        confidence = {'data': df['confidence'].values.tolist()}
        confidence.update(get_coloring_info(df['confidence']))

    class_info = {'class': predictions, 'confidence': confidence}

    monitoring_data = {}
    for (monitoring_metric_name, comparison_operator) in monitoring_fields:
        if monitoring_metric_name in df.columns:
            scores = np.array(df[monitoring_metric_name])
            thresholds = np.array(df[f'{monitoring_metric_name}_threshold'])

            monitoring_data[monitoring_metric_name] = {'scores': scores.tolist(), 'threshold': thresholds[0],
                                                       # taking last threshold
                                                       'operation': comparison_operator}
            monitoring_data[monitoring_metric_name].update(get_coloring_info(df[monitoring_metric_name]))

    return {'class_labels': class_info,
            'metrics': monitoring_data,
            'requests_ids': requests_ids,
            'top_100': top_100_neighbours,
            'counterfactuals': counterfactuals}


def parse_embeddings_from_dataframe(df):
    embeddings = np.apply_along_axis(lambda x: np.array(x), arr=df['embedding'].values.tolist(), axis=0)
    return embeddings


def get_record(db, method, model_name, model_version) -> Dict:
    existing_record = db[method].find_one({"model_name": model_name,
                                           "model_version": model_version})
    if not existing_record:
        return {"model_name": model_name,
                "model_version": model_version,
                "embeddings_bucket_name": "",
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


def compute_training_embeddings(model: Model, servable: HydroServingServable,
                                training_data: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Computes embeddings from training data using model servable
    :param model: instance of HydroServingModel
    :param servable: instance of HydroServingServable
    :param training_data: Dataframe with training data
    :return: np.ndarray with embeddings or None if failed
    """
    input_names = list(map(lambda x: x.name, model.contract.predict.inputs))
    if len(set(input_names) - set(training_data.columns)) != 0:
        raise ValueError(
            f'Input fields ({input_names}) are not compatible with data fields ({set(training_data.columns)})')
    inputs = training_data[input_names]
    embeddings = []
    logging.debug('Embedding inference: ')
    for i in tqdm(range(len(inputs))):
        outputs = servable(inputs.iloc[i])
        embeddings.append(outputs['embedding'])

    return np.concatenate(embeddings)
