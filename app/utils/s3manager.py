import logging
import json
import tempfile
from typing import Optional, Any

import pandas as pd
import s3fs
import boto3
import joblib
from botocore.client import Config


class S3Manager:
    def __init__(self, bucket_name: str, aws_storage_endpoint: Optional[str], aws_region: Optional[str], ):
        if aws_storage_endpoint:
            client_kwargs = {
                'endpoint_url': aws_storage_endpoint,
                'region_name': aws_region,
            }
            self.fs = s3fs.S3FileSystem(
                anon=False,
                client_kwargs=client_kwargs,
                config_kwargs={'s3': {'addressing_style': 'path'}}
            )
            conf = Config(s3={'addressing_style': 'path'})
            boto_client = boto3.client('s3', config=conf, **client_kwargs)
        else:
            self.fs = s3fs.S3FileSystem()
            boto_client = boto3.client('s3')

        if not self.fs.exists(f's3://{bucket_name}'):
            logging.info(f'Creating {bucket_name} bucket')
            try:
                boto_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={
                        'LocationConstraint': aws_region
                    }
                )
            except Exception as e:
                if not self.fs.exists(f's3://{bucket_name}'):
                    logging.error(f"Couldn't create {bucket_name} bucket due to error: {e}")
                    raise ValueError(f"Couldn't create {bucket_name} bucket due to error: {e}") from e

    def write_parquet(self, df: pd.DataFrame, bucket_name, filename):
        try:
            df.to_parquet(f's3://{bucket_name}/{filename}')
        except Exception as e:
            logging.error(f'Couldn\'t write {filename} to {bucket_name} bucket. Error: {e}')

    def write_json(self, data: dict, filepath: str):
        with self.fs.open(filepath, 'w') as f:
            f.write(json.dumps(data))

    def read_json(self, filepath: str) -> dict:
        '''
        Reads json
        If file not found or couldn't parse content, return empty Dict and log error
        :param filepath: in format 's3://path-to-file'
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

    def dump_with_joblib(self, object, filepath) -> bool:
        """
        :param object:
        :param filepath: in format 's3://path-to-file'
        :return: success
        """
        with tempfile.TemporaryFile() as fp:
            try:
                joblib.dump(object, fp)
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

    def load_with_joblib(self, filepath) -> Optional[Any]:
        """
        Loads joblib objects from S3
        if file not found: return None
        if is not joblib file: return None
        :param filepath: in format 's3://path-to-file'
        :return: Object/None
        """
        clean_path = filepath.replace('s3://', '')
        if not self.fs.exists(filepath):
            logging.error(f'No such file {filepath}')
            return None
        with self.fs.open(clean_path, 'rb') as f:
            try:
                loaded_object = joblib.load(f)
            except Exception as e:
                logging.error(f'Couldn\'t load joblib file: {filepath}. Error: {e}')
                return None
            return loaded_object

