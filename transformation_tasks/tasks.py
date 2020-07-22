import json
import sys
from datetime import datetime
from typing import List

import hydro_serving_grpc as hs_grpc
import pandas as pd
import requests
from celery.exceptions import Ignore
from hydrosdk.cluster import Cluster
from hydrosdk.modelversion import ModelVersion
from hydrosdk.servable import Servable
from loguru import logger as logging

from app import celery, s3manager
from ml_transformers.transformer import transform_high_dimensional
from ml_transformers.utils import VisMetrics, DEFAULT_PROJECTION_PARAMETERS
from utils.conf import MONGO_URL, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_AUTH_DB, HYDRO_VIS_BUCKET_NAME, \
    EMBEDDING_FIELD, HS_CLUSTER_ADDRESS, GRPC_PROXY_ADDRESS, TaskStates
from utils.data_management import get_record, parse_embeddings_from_dataframe, parse_requests_dataframe, \
    update_record, get_mongo_client, get_production_subsample, compute_training_embeddings, valid_embedding_model


def get_training_data_path(model: ModelVersion) -> str:
    """

    :param model:
    :return:
    """
    response = requests.get(f'{HS_CLUSTER_ADDRESS}/monitoring/training_data?modelVersionId={model.id}')
    training_data_s3 = json.loads(response.text)
    if training_data_s3:
        return training_data_s3[0]
    else:
        return ''


def get_production_data_sample(model_id, sample_size=1000) -> pd.DataFrame:
    response = requests.get(f'{HS_CLUSTER_ADDRESS}/monitoring/checks/subsample/{model_id}?size={sample_size}')
    return pd.DataFrame.from_dict(response.json())


@celery.task(bind=True, track_started=True)
def transform_task(self, method, model_version_id):
    start = datetime.now()
    mongo_client = get_mongo_client(MONGO_URL, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_AUTH_DB)
    db = mongo_client['visualization']

    s3_model_path = f's3://{HYDRO_VIS_BUCKET_NAME}/{model_version_id}'  # TODO make management of S3 bucket storage to check if model in storage is correct
    s3manager.fs.mkdirs(s3_model_path, exist_ok=True)

    db_model_info = get_record(db, method, model_version_id)
    parameters = db_model_info.get('parameters', {})
    path_to_transformer = db_model_info.get('transformer_file', '')
    result_path = db_model_info.get('result_file', '')
    vis_metrics: List[str] = db_model_info.get('visualization_metrics',
                                               DEFAULT_PROJECTION_PARAMETERS['visualization_metrics'])
    vis_metrics: List[VisMetrics] = [VisMetrics(metric_name) for metric_name in vis_metrics]
    training_data_sample_size = db_model_info.get('training_data_sample_size',
                                                  DEFAULT_PROJECTION_PARAMETERS['training_data_sample_size'])

    production_data_sample_size = db_model_info.get('production_data_sample_size',
                                                    DEFAULT_PROJECTION_PARAMETERS['production_data_sample_size'])
    if result_path:
        plottable_data = s3manager.read_json(filepath=result_path)
        if plottable_data:
            return {"result": plottable_data}, 200

    try:
        logging.info(f'Connecting to cluster')
        hs_cluster = Cluster(HS_CLUSTER_ADDRESS, grpc_address=GRPC_PROXY_ADDRESS)
        model = ModelVersion.find_by_id(hs_cluster, int(model_version_id))
        model_name = model.name
        model_version = model.version
    except ValueError as e:
        self.update_state(state=TaskStates.ERROR,
                          meta={'message': f"Error: {e}", 'code': 404})
        raise Ignore()
    except Exception as e:
        self.update_state(state=TaskStates.ERROR,
                          meta={'message': f"Error: {e}", 'code': 500})
        raise Ignore()
    if not valid_embedding_model(model):
        self.update_state(state=TaskStates.NOT_SUPPORTED,
                          meta={'message': f"Invalid model {model} contract: No {EMBEDDING_FIELD} field in outputs",
                                'code': 404})
        raise Ignore()
    path_to_training_data = get_training_data_path(model)

    if path_to_transformer:
        transformer = s3manager.read_transformer_model(filepath=path_to_transformer)
    else:
        transformer = None

    # Parsing model requests and training data
    if path_to_training_data:
        try:
            training_df = pd.read_csv(s3manager.fs.open(path_to_training_data,
                                                        mode='rb'))
        except:
            e = sys.exc_info()[0]
            logging.error(f'Couldn\'t get training data from {path_to_training_data}: {e}')
            training_df = None
    else:
        training_df = None
    production_requests_df = get_production_subsample(model.id, production_data_sample_size)

    if production_requests_df.empty:
        self.update_state(state=TaskStates.NO_DATA,
                          meta={'message': f'{model_name}v{model_version} model has not enough production data.',
                                'code': 404})
        raise Ignore()
    if EMBEDDING_FIELD not in production_requests_df.columns:
        self.update_state(state=TaskStates.ERROR,
                          meta={'message': f"Unable to get requests embeddings of {model_name}v{model_version} model",
                                'code': 404})
        raise Ignore()

    production_embeddings = parse_embeddings_from_dataframe(production_requests_df)

    requests_data_dict = parse_requests_dataframe(production_requests_df, monitoring_models_conf, production_embeddings)

    logging.info(f'Parsed requests data shape: {production_embeddings.shape}')

    if training_df is None:
        training_embeddings = None
    elif EMBEDDING_FIELD in training_df.columns:
        logging.debug('Training embeddings exist')
    else:
        try:
            logging.info('Creating servable')
            manager_stub = hs_grpc.manager.ManagerServiceStub(channel=hs_cluster.channel)
            deploy_request = hs_grpc.manager.DeployServableRequest(version_id=model.id,
                                                                   metadata={"created_by": "hydro_vis"})
            for servable_proto in manager_stub.DeployServable(deploy_request):
                logging.info(f"{servable_proto.name} is {servable_proto.status}")
            if servable_proto.status != 3:
                raise ValueError(f"Invalid servable state came from GRPC stream - {servable_proto.status}")
            servable = Servable.find(hs_cluster, servable_name=servable_proto.name)

        except Exception as e:
            logging.error(f"Couldn't create {model_name}v{model_version} servable. Error:{e}")
            training_embeddings = None
        else:
            training_data_sample_size = min(training_data_sample_size, len(training_df))
            training_df_sample = training_df.sample(training_data_sample_size)
            training_embeddings = compute_training_embeddings(model, servable, training_df_sample)
            Servable.delete(hs_cluster, servable.name)

    plottable_data, transformer = transform_high_dimensional(method, parameters,
                                                             training_embeddings, production_embeddings,
                                                             transformer,
                                                             vis_metrics=vis_metrics)
    plottable_data.update(requests_data_dict)

    result_path = s3_model_path + '/result.json'
    try:
        s3manager.write_json(data=plottable_data, filepath=result_path)
        db_model_info["result_file"] = result_path
    except:
        e = sys.exc_info()[1]
        logging.error(f'Couldn\'t save result to {result_path}: {e}')

    transformer_path = s3_model_path + f'/transformer_{method}_{model_name}{model_version}'
    try:
        transformer_saved_to_s3 = s3manager.write_transformer_model(transformer,
                                                                    filepath=transformer_path)
    except:
        e = sys.exc_info()[1]
        logging.error(f'Couldn\'t save transformer to {transformer_path}: {e}')
        transformer_saved_to_s3 = False

    if transformer_saved_to_s3:
        db_model_info['transformer_file'] = transformer_path

    update_record(db, method, db_model_info, model.id)

    logging.info(f'Request handled in {datetime.now() - start}')

    return {"result": plottable_data}, 200
