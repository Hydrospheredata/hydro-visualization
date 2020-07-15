import sys
from datetime import datetime
from typing import List, Optional, Tuple

import hydro_serving_grpc as hs_grpc
import numpy as np
import pandas as pd
from celery.exceptions import Ignore
from hydrosdk.cluster import Cluster
from hydrosdk.modelversion import ModelVersion
from hydrosdk.monitoring import MetricSpec
from hydrosdk.servable import Servable
from loguru import logger as logging

from app import celery, s3manager
from ml_transformers.autoembeddings import AutoEmbeddingsEncoder, dataframe_to_feature_map, TransformationType, \
    NUMERICAL_TRANSFORMS
from ml_transformers.transformer import transform_high_dimensional, transform_high_dimensional_mixed, \
    UmapTransformerWithMixedTypes
from ml_transformers.utils import VisMetrics, DEFAULT_PROJECTION_PARAMETERS
from utils.conf import MONGO_URL, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_AUTH_DB, HYDRO_VIS_BUCKET_NAME, \
    HS_CLUSTER_ADDRESS, GRPC_PROXY_ADDRESS, TaskStates
from utils.data_management import get_record, parse_embeddings_from_dataframe, parse_requests_dataframe, \
    update_record, get_mongo_client, get_production_subsample, compute_training_embeddings, model_has_embeddings, \
    calcualte_neighbours, get_training_data_path


def get_embeddings(production_df: pd.DataFrame, training_df: pd.DataFrame, training_data_sample_size: int,
                   hs_cluster: Cluster, model: ModelVersion):
    production_embeddings = parse_embeddings_from_dataframe(production_df)
    if training_df is None:
        training_embeddings = None
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
            logging.error(f"Couldn't create {model.name}v{model.version} servable. Error:{e}")
            training_embeddings = None
        else:
            training_data_sample_size = min(training_data_sample_size, len(training_df))
            training_df_sample = training_df.sample(training_data_sample_size)
            training_embeddings = compute_training_embeddings(model, servable, training_df_sample)
            Servable.delete(hs_cluster, servable.name)

    return production_embeddings, training_embeddings


def generate_auto_embeddings(production_data: pd.DataFrame, training_data: pd.DataFrame, model: ModelVersion,
                             encoder: Optional[AutoEmbeddingsEncoder]) -> Tuple[
    List[np.array], List[np.array], AutoEmbeddingsEncoder]:
    """

    :param training_data:
    :param production_data:
    :param model:
    :param encoder:
    :return:
    """
    model_inputs = list(model.contract.predict.inputs)
    input_names = [model_input.name for model_input in model_inputs]
    used_inputs = training_data.columns.intersection(input_names)
    training_feature_map = dataframe_to_feature_map(training_data[used_inputs], model)
    production_feature_map = dataframe_to_feature_map(production_data[used_inputs], model)
    if len(training_feature_map) == 0 or len(production_feature_map) == 0:
        return None, None  # not enough data

    if encoder is None:
        encoder = AutoEmbeddingsEncoder()
        training_embeddings_map = encoder.fit_transform(training_feature_map)
    else:
        training_embeddings_map = encoder.transform(training_feature_map)

    production_embeddings_map = encoder.transform(production_feature_map)
    training_numerical_embeddings, training_categorical_embeddings = None, None
    production_numerical_embeddings, production_categorical_embeddings = None, None
    if TransformationType.ONE_HOT in training_embeddings_map.keys():
        training_categorical_embeddings = np.concatenate(
            [v for k, v in training_embeddings_map.items() if k == TransformationType.ONE_HOT], axis=1)
        production_categorical_embeddings = np.concatenate(
            [v for k, v in production_embeddings_map.items() if k == TransformationType.ONE_HOT], axis=1)
    if len(NUMERICAL_TRANSFORMS & set(training_embeddings_map.keys())) != 0:
        training_numerical_embeddings = np.concatenate(
            [v for k, v in training_embeddings_map.items() if k != TransformationType.ONE_HOT], axis=1)
        production_numerical_embeddings = np.concatenate(
            [v for k, v in production_embeddings_map.items() if k != TransformationType.ONE_HOT], axis=1)

    return [production_numerical_embeddings, production_categorical_embeddings], [training_numerical_embeddings,
                                                                                  training_categorical_embeddings], encoder


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
    path_to_result_file = db_model_info.get('result_file', '')
    path_to_encoder = db_model_info.get('encoder_file', '')

    if path_to_result_file:
        plottable_result = s3manager.read_json(filepath=path_to_result_file)
        if plottable_result:
            return {"result": plottable_result}, 200

    vis_metrics: List[str] = db_model_info.get('visualization_metrics',
                                               DEFAULT_PROJECTION_PARAMETERS['visualization_metrics'])
    vis_metrics: List[VisMetrics] = [VisMetrics.to_enum(metric_name) for metric_name in vis_metrics]

    training_data_sample_size = db_model_info.get('training_data_sample_size',
                                                  DEFAULT_PROJECTION_PARAMETERS['training_data_sample_size'])

    production_data_sample_size = db_model_info.get('production_data_sample_size',
                                                    DEFAULT_PROJECTION_PARAMETERS['production_data_sample_size'])

    try:
        hs_cluster = Cluster(HS_CLUSTER_ADDRESS, grpc_address=GRPC_PROXY_ADDRESS)
        model = ModelVersion.find_by_id(hs_cluster, int(model_version_id))
        model_name = model.name
        model_version = model.version
        embeddings_exist = model_has_embeddings(model)
    except ValueError as e:
        self.update_state(state=TaskStates.ERROR,
                          meta={'message': f"Error: {e}", 'code': 404})
        raise Ignore()
    except Exception as e:
        self.update_state(state=TaskStates.ERROR,
                          meta={'message': f"Error: {e}", 'code': 500})
        raise Ignore()

    if path_to_transformer:
        transformer = s3manager.load_with_joblib(filepath=path_to_transformer)
    else:
        transformer = None

    if path_to_encoder:
        autoembeddings_encoder = s3manager.load_with_joblib(filepath=path_to_encoder)
    else:
        autoembeddings_encoder = None

    # Parsing model requests and training data
    path_to_training_data = get_training_data_path(model)
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
    if ((training_df is None) or training_df.empty) and (not embeddings_exist):
        self.update_state(state=TaskStates.NO_DATA,
                          meta={
                              'message': f'{model_name}v{model_version} model requires training data to generate autoembeddings.',
                              'code': 404})
        raise Ignore()
    if embeddings_exist:
        production_embeddings, training_embeddings = get_embeddings(production_requests_df, training_df,
                                                                    training_data_sample_size, hs_cluster, model)
    else:
        [production_numerical_embeddings, production_categorical_embeddings], \
        [training_numerical_embeddings, training_categorical_embeddings], \
        autoembeddings_encoder = generate_auto_embeddings(production_requests_df, training_df, model,
                                                          encoder=autoembeddings_encoder)

        if training_numerical_embeddings is not None and training_categorical_embeddings is None:  # Only numerical features are used
            production_embeddings, training_embeddings = production_numerical_embeddings, training_numerical_embeddings
        elif training_categorical_embeddings is not None and training_numerical_embeddings is None:  # Only categorical features are used
            production_embeddings, training_embeddings = production_categorical_embeddings, training_categorical_embeddings
            parameters['metric'] = 'dice'
        elif training_categorical_embeddings is not None and training_numerical_embeddings is not None:  # Both
            production_embeddings, training_embeddings = [production_numerical_embeddings,
                                                          production_categorical_embeddings], \
                                                         [training_numerical_embeddings,
                                                          training_categorical_embeddings]
        else:
            self.update_state(state=TaskStates.NOT_SUPPORTED,
                              meta={'message': f"Couldn\'t extract autoembeddings from {model}.",
                                    'code': 404})
            raise Ignore()

    monitoring_models_conf = [(metric.name, metric.config.threshold_op, metric.config.threshold) for metric in
                              MetricSpec.list_for_model(hs_cluster, model.id)]

    if isinstance(training_embeddings, np.ndarray):  # not mixed-type data
        top_N_neighbours = calcualte_neighbours(production_embeddings)
        plottable_result, transformer = transform_high_dimensional(method, parameters,
                                                                   training_embeddings, production_embeddings,
                                                                   transformer,
                                                                   vis_metrics=vis_metrics)
    else:  # mixed-type data
        top_N_neighbours = []
        plottable_result, transformer = transform_high_dimensional_mixed(method, parameters,
                                                                         training_embeddings, production_embeddings)

    requests_data_dict = parse_requests_dataframe(production_requests_df, monitoring_models_conf,
                                                  top_N_neighbours)
    plottable_result.update(requests_data_dict)

    path_to_result_file = s3_model_path + '/result.json'
    try:
        s3manager.write_json(data=plottable_result, filepath=path_to_result_file)
        db_model_info["result_file"] = path_to_result_file
    except:
        e = sys.exc_info()[1]
        logging.error(f'Couldn\'t save result to {path_to_result_file}: {e}')

    if type(transformer) != UmapTransformerWithMixedTypes:  # MixedTypes is used only with fit_transform
        transformer_path = s3_model_path + f'/transformer_{method}_{model_name}{model_version}'
        try:
            transformer_saved_to_s3 = s3manager.dump_with_joblib(transformer,
                                                                 filepath=transformer_path)
        except:
            e = sys.exc_info()[1]
            logging.error(f'Couldn\'t save transformer to {transformer_path}: {e}')
            transformer_saved_to_s3 = False

        if transformer_saved_to_s3:
            db_model_info['transformer_file'] = transformer_path
    else:
        autoembeddings_encoder_path = s3_model_path + f'/autoembeddings_encoder_{method}_{model_name}{model_version}'
        try:
            encoder_saved_to_s3 = s3manager.dump_with_joblib(autoembeddings_encoder, filepath=autoembeddings_encoder_path)
        except:
            e = sys.exc_info()[1]
            logging.error(f'Couldn\'t save transformer to {autoembeddings_encoder_path}: {e}')
            encoder_saved_to_s3 = False
        if encoder_saved_to_s3:
            db_model_info['encoder_file'] = autoembeddings_encoder_path

    update_record(db, method, db_model_info, model.id)

    logging.info(f'Request handled in {datetime.now() - start}')

    return {"result": plottable_result}, 200
