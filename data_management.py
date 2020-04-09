import json
import math
import random
import tempfile
from typing import Dict, Optional, List, Tuple

import fastparquet
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
from conf import AWS_STORAGE_ENDPOINT, FEATURE_LAKE_BUCKET, BATCH_SIZE
from ml_transformers.transformer import Transformer
from ml_transformers.utils import DEFAULT_PARAMETERS, Coloring, get_top_100_neighbours


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
                }
            )
        else:
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

    def write_json(self, data: Dict, filepath:str):
        with self.fs.open(filepath, 'w') as f:
            f.write(json.dumps(data))

    def read_json(self, filepath:str) -> Dict:
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

    def get_production_subsample(self, model: Model, size: int, undersampling = False) -> pd.DataFrame:
        """
        Return a random subsample of request-response pairs from an S3 feature lake.


        :param undersampling: If True, returns subsample of size = min(available samples, size),
        if False and number of samples stored in feature lake < size then raise an Exception
        :param size: Number of requests\response pairs in subsample
        :type batch_size: Number of requests stored in each parquet file
        """

        number_of_parquets_needed = math.ceil(size / BATCH_SIZE)

        model_feature_store_path = f"s3://{FEATURE_LAKE_BUCKET}/{model.name}/{model.version}"

        parquets_paths = self.fs.find(model_feature_store_path)

        if len(parquets_paths) < number_of_parquets_needed:
            if not undersampling:
                raise ValueError(
                    f"This model doesn't have {size} requests in feature lake.\n"
                    f"Right now there are {len(parquets_paths) * BATCH_SIZE} requests stored.")
            else:
                number_of_parquets_needed = len(parquets_paths)

        selected_batch_paths = random.sample(parquets_paths, number_of_parquets_needed)

        dataset = fastparquet.ParquetFile(selected_batch_paths, open_with=self.fs.open)

        df: pd.DataFrame = dataset.to_pandas()

        if df.shape[0] > size:
            return df.sample(n=size)
        else:
            return df


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
    # for (monitoring_metric_name, comparison_operator) in monitoring_fields:
    #     if monitoring_metric_name in df.columns:
    #         scores = np.array(df[monitoring_metric_name])
    #         thresholds = np.array(df[f'{monitoring_metric_name}_threshold'])
    #
    #         monitoring_data[monitoring_metric_name] = {'scores': scores.tolist(), 'threshold': thresholds[0],
    #                                                    # taking last threshold
    #                                                    'operation': comparison_operator}
    #         monitoring_data[monitoring_metric_name].update(get_coloring_info(df[monitoring_metric_name]))

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
