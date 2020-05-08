import json
import sys
from datetime import datetime
from time import sleep

import grpc
import pandas as pd
import requests
from hydrosdk.exceptions import ServableException
from hydrosdk.cluster import Cluster
from hydrosdk.model import Model
from hydrosdk.monitoring import MetricSpec
from hydrosdk.servable import Servable, ServableStatus
from loguru import logger as logging
import hydro_serving_grpc as hs_grpc

from app import celery, s3manager
from conf import MONGO_URL, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_AUTH_DB, HYDRO_VIS_BUCKET_NAME, \
    EMBEDDING_FIELD, HS_CLUSTER_ADDRESS, GRPC_UI_ADDRESS
from data_management import get_record, parse_embeddings_from_dataframe, parse_requests_dataframe, \
    update_record, get_mongo_client, get_production_subsample, compute_training_embeddings
from visualizer import transform_high_dimensional


def valid_embedding_model(model: Model) -> [bool]:
    """
    TODO add embedding field shape check
    Check if model returns embeddings
    :param model:
    :return:
    """
    output_names = [field.name for field in model.contract.predict.outputs]
    if EMBEDDING_FIELD not in output_names:
        return False
    return True


def get_training_data_path(model: Model) -> str:
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


@celery.task(bind=True)
def transform_task(self, method, request_json):
    start = datetime.now()
    mongo_client = get_mongo_client(MONGO_URL, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_AUTH_DB)
    db = mongo_client['visualization']

    model_name = request_json.get('model_name')
    model_version = str(request_json.get('model_version'))
    vis_metrics = request_json.get('visualization_metrics', {})

    s3_model_path = f's3://{HYDRO_VIS_BUCKET_NAME}/{model_name}/{model_version}'
    s3manager.fs.mkdirs(s3_model_path, exist_ok=True)

    db_model_info = get_record(db, method, model_name, str(model_version))
    parameters = db_model_info.get('parameters', {})
    path_to_transformer = db_model_info.get('transformer_file', '')
    result_path = db_model_info.get('result_file', '')
    if result_path:
        plottable_data = s3manager.read_json(filepath=result_path)
        if plottable_data:
            return {"result": plottable_data}, 200

    try:
        hs_cluster = Cluster(HS_CLUSTER_ADDRESS, grpc_address=GRPC_UI_ADDRESS)
        model = Model.find(hs_cluster, model_name, int(model_version))
    except ValueError as e:
        return {"message": f"Unable to find {model_name}v{model_version}. Error: {e}"}, 404
    except Exception as e:
        return {"message": f"Error {model_name}v{model_version}. Error: {e}"}, 500
    if not valid_embedding_model(model):
        return {"message": f"Invalid model {model} contract: No {EMBEDDING_FIELD} field in outputs"}, 404
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
    production_requests_df = get_production_subsample(model.id, 200)

    if production_requests_df.empty:
        return f'Production data is empty', 404
    if EMBEDDING_FIELD not in production_requests_df.columns:
        return "Unable to get requests embeddings", 404

    production_embeddings = parse_embeddings_from_dataframe(production_requests_df)
    monitoring_models_conf = [(metric.name, metric.config.threshold_op, metric.config.threshold) for metric in
                              MetricSpec.list_for_model(hs_cluster, model.id)]
    requests_data_dict = parse_requests_dataframe(production_requests_df, monitoring_models_conf, production_embeddings)

    logging.info(f'Parsed requests data shape: {production_embeddings.shape}')

    if training_df is None:
        training_embeddings = None
    elif EMBEDDING_FIELD in training_df.columns:
        logging.debug('Training embeddings exist')
    else:
        try:
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
            training_embeddings = compute_training_embeddings(model, servable, training_df)
            Servable.delete(hs_cluster, servable.name)

    plottable_data, transformer = transform_high_dimensional(method, parameters,
                                                             training_embeddings, production_embeddings,
                                                             transformer,
                                                             vis_metrics=vis_metrics)
    plottable_data.update(requests_data_dict)
    plottable_data["parameters"] = parameters

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

    update_record(db, method, db_model_info, model_name, model_version)

    logging.info(f'Request handled in {datetime.now() - start}')

    return {"result": plottable_data}, 200
