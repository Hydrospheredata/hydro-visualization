import json
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from hydrosdk.model import Model
from hydrosdk.monitoring import MetricSpec
from loguru import logger as logging

from app import celery, s3manager, hs_cluster, hs_client
from conf import MONGO_URL, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_AUTH_DB, CLUSTER_URL, HYDRO_VIS_BUCKET_NAME
from data_management import get_record, parse_embeddings_from_dataframe, parse_requests_dataframe, \
    update_record, get_mongo_client, compute_training_embeddings, get_production_subsample
from visualizer import transform_high_dimensional


def valid_embedding_model(model: Model) -> [bool]:
    """
    Check if model returns embeddings
    :param model:
    :return:
    """
    output_names = [field.name for field in model.contract.predict.outputs]
    if 'embedding' not in output_names:
        return False
    return True


def get_training_data_path(model: Model) -> str:
    """

    :param model:
    :return:
    """
    response = requests.get(f'{CLUSTER_URL}/monitoring/training_data?modelVersionId={model.id}')
    training_data_s3 = json.loads(response.text)
    if training_data_s3:
        return training_data_s3[0]
    else:
        return ''


def get_production_data_sample(model_id, sample_size=1000) -> pd.DataFrame:
    response = requests.get(f'{CLUSTER_URL}/monitoring/checks/subsample/{model_id}?size={sample_size}')
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
        model = Model.find(hs_cluster, model_name, int(model_version))
    except ValueError as e:
        return {"message": f"Unable to find {model_name}v{model_version}. Error: {e}"}, 404
    except Exception as e:
        return {"message": f"Error {model_name}v{model_version}. Error: {e}"}, 500
    if not valid_embedding_model(model):
        return {"message": f"Invalid model {model} contract: No 'embedding' field in outputs"}, 404
    path_to_training_data = get_training_data_path(model)

    if path_to_transformer:
        transformer = s3manager.read_transformer_model(filepath=path_to_transformer)
    else:
        transformer = None

    # Parsing model requests and training data
    if path_to_training_data:
        training_df = pd.read_csv(path_to_training_data)
    else:
        training_df = None
    production_requests_df = get_production_subsample(model.id, 1000)

    if production_requests_df.empty:
        return f'Production data is empty', 404
    if 'embedding' not in production_requests_df.columns:
        return "Unable to get requests embeddings", 404

    production_embeddings = parse_embeddings_from_dataframe(production_requests_df)
    monitoring_models_conf = [(metric.name, metric.config.threshold_op, metric.config.threshold) for metric in
                              MetricSpec.list_for_model(hs_cluster, model.id)]
    requests_data_dict = parse_requests_dataframe(production_requests_df, monitoring_models_conf, production_embeddings)

    logging.info(f'Parsed requests data shape: {production_embeddings.shape}')

    if training_df is None:
        training_embeddings = None
    elif 'embedding' in training_df.columns:
        logging.debug('Training embeddings exist')
        training_embeddings = np.stack(training_df['embedding'].values)
    else:  # infer embeddings using model
        try:
            servable = hs_client.deploy_servable(model_name, int(model_version))  # TODO SDK
        except ValueError as e:
            return {"message": f"Unable to find {model_name}v{model_version}. Error: {e}"}, 404
        except Exception as e:
            return {"message": f"Error {model_name}v{model_version}. Error: {e}"}, 500

        training_embeddings = compute_training_embeddings(model, servable, training_df)
        servable.delete()

    plottable_data, transformer = transform_high_dimensional(method, parameters,
                                                             training_embeddings, production_embeddings,
                                                             transformer,
                                                             vis_metrics=vis_metrics)
    plottable_data.update(requests_data_dict)
    plottable_data["parameters"] = parameters

    result_path = s3_model_path + '/result.json'
    s3manager.write_json(data=plottable_data, filepath=result_path)
    db_model_info["result_file"] = result_path

    transformer_path = s3_model_path + f'/transformer_{method}_{model_name}{model_version}'
    transformer_saved_to_s3 = s3manager.write_transformer_model(transformer,
                                                                filepath=transformer_path)

    if transformer_saved_to_s3:
        db_model_info['transformer_file'] = transformer_path

    update_record(db, method, db_model_info, model_name, model_version)

    logging.info(f'Request handled in {datetime.now() - start}')

    return {"result": plottable_data}, 200
