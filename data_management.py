import json
import tempfile
from enum import Enum
from typing import Dict, Optional, List, Tuple

import boto3
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import s3fs
from loguru import logger as logging
from tqdm import tqdm

from client import HydroServingModel, HydroServingServable
from ml_transformers.transformer import Transformer
from ml_transformers.utils import DEFAULT_PARAMETERS


class S3Manager:
    def __init__(self):
        self.s3 = boto3.client('s3')
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

    def write_json(self, data: Dict, bucket_name: str, filename: str) -> bool:
        try:
            self.s3.put_object(Bucket=bucket_name, Body=json.dumps(data), Key=filename)
        except Exception as e:
            logging.error(f'Couldn\'t write json to {bucket_name} bucket. Error: {e}')
            return False
        return True

    def read_json(self, bucket_name: str, filename: str) -> Dict:
        if not self.file_exists(bucket_name, filename):
            return {}
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
        """
        Loads pretrained transformer model from S3
        :param bucket_name:
        :param filename:
        :return: Transformer/None
        """
        if not self.file_exists(bucket_name, filename):
            logging.error('Transformer doesn\'t exists')
            return None
        try:
            with tempfile.TemporaryFile() as fp:
                self.s3.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=filename)
                fp.seek(0)
                transformer = joblib.load(fp)
                return transformer
        except Exception as e:
            logging.error(f'Couldn\'t read transformer from {bucket_name}/{filename} bucket. Error: {e}')


def parse_requests_dataframe(df, monitoring_fields: List[Tuple[str, str]]) -> (Dict, np.ndarray):
    """
    Extracts:
        - embeddings,
        - class prediction,
        - confidence,
        - other monitoring metrics and thresholds
    from requests dataframe
    :param df: dataframe
    :param monitoring_fields: list of monitoring metrics names with comparison operator
    :return: (Dict {"class_labels":{…}, "metrics":{…}}, request embeddings)
    """
    monitoring = {}
    predictions, confidence = [], []
    if 'class' in df.columns:
        predictions = df['class'].values.tolist()
    if 'confidence' in df.columns:
        confidence = df['confidence'].values.tolist()
    requests_ids = df['request_id'].values.tolist()
    embeddings = np.apply_along_axis(lambda x: np.array(x), arr=df['embedding'].values.tolist(), axis=0)
    for (monitoring_name, comp_operator) in monitoring_fields:
        if monitoring_name in df.columns:
            scores = np.array(df[monitoring_name])
            thresholds = np.array(df[f'{monitoring_name}_threshold'])
            monitoring[monitoring_name] = {'scores': scores.tolist(), 'threshold': thresholds[0],
                                           'operation': comp_operator}  # taking last threshold
    class_info = {'class': predictions, 'confidence': confidence}
    return {'class_labels': class_info, 'metrics': monitoring, 'requests_ids': requests_ids}, embeddings


def get_requests_data(requests_df: pd.DataFrame, monitoring_fields) -> (Dict, np.ndarray):
    """
    :param requests_df: dataframe with model requests file
    :param monitoring_fields:
    :return: coloring info, requests embeddings
    """
    if not requests_df:
        return None, None

    if 'embedding' not in requests_df.columns:
        logging.error(f'No "embedding" field')
        return None, None

    return parse_requests_dataframe(requests_df, monitoring_fields)


class RecordStatus(Enum):
    CREATED = 1
    EXISTED = 2
    MODIFIED = 3


def get_record(db, method, model_name, model_version) -> [Dict, RecordStatus]:
    existing_record = db[method].find_one({"model_name": model_name,
                                           "model_version": model_version})
    if existing_record is None:
        return {"model_name": model_name,
                "model_version": model_version,
                "embeddings_bucket_name": "",
                "result_file": "",
                "transformer_file": "",
                "parameters": DEFAULT_PARAMETERS[method],
                "use_labels": False}, RecordStatus.CREATED
    return existing_record, RecordStatus.EXISTED


def save_record(db, method: str, record: Dict):
    """
    Saves information about transformer to database or updates information if it existed
    :param db:
    :param method: 'umap' (method name from ml_transformers.utils.AVAILABLE_TRANSFORMERS)
    :param record: Dict with values
    :return:
    """
    model_name = record['model_name']
    model_version = record['model_version']
    existing_record, status = get_record(db, method, model_name, model_version)

    if status == RecordStatus.EXISTED:
        db[method].update_one({"model_name": model_name,
                               "model_version": model_version}, {"$set": record})

    else:
        db[method].insert_one(record)


def save_model_params(db, model_name: str, model_version: str, method: str, parameters: Dict,
                      use_labels: bool) -> RecordStatus:
    """
    Saves model parameters for transformer method
    :param db: DataBase object
    :param model_name:
    :param model_version:
    :param method: transformer method
    :param parameters: dict with transformer parameters
    :param use_labels:
    :return: RecordStatus
    """
    if not parameters:
        parameters = DEFAULT_PARAMETERS[method]

    record, status = get_record(db, method, model_name, model_version)
    if parameters == record.get("parameters", {}) & use_labels == record.get("use_labels",
                                                                             False) & status == RecordStatus.EXISTED:
        logging.info("Transformer with same parameters already existed")
        return status
    else:
        record["parameters"] = parameters
        record["use_labels"] = use_labels
        save_record(db, method, record)
        return RecordStatus.MODIFIED


def get_training_embeddings(model: HydroServingModel, servable: HydroServingServable,
                            training_data: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Computes embeddings from training data using model servable
    :param model: instance of HydroServingModel
    :param servable: instance of HydroServingServable
    :param training_data: Dataframe with training data
    :return: np.ndarray with embeddings or None if failed
    """
    if not training_data:
        return None
    if 'embedding' in training_data.columns:
        logging.debug('Training embeddings exist')
        embs = np.stack(training_data['embedding'].values)
        return embs

    input_names = list(map(lambda x: x['name'], model.contract.contract_dict['inputs']))
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
