import logging
from re import I
import sys
from collections import namedtuple
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from celery.exceptions import Ignore
from hydrosdk.cluster import Cluster
from hydrosdk.modelversion import ModelVersion
from hydrosdk.servable import Servable

from hydro_viz.celery_app import celery_app
from hydro_viz.utils.conf import s3manager, get_mongo_client, get_hs_cluster
from hydro_viz.ml_transformers.autoembeddings import AutoEmbeddingsEncoder, dataframe_to_feature_map, TransformationType, \
    NUMERICAL_TRANSFORMS
from hydro_viz.ml_transformers.transformer import transform_high_dimensional, transform_high_dimensional_mixed, \
    UmapTransformerWithMixedTypes
from hydro_viz.ml_transformers.utils import VisMetrics, DEFAULT_PROJECTION_PARAMETERS
from hydro_viz.utils.conf import (
    HYDRO_VIS_BUCKET_NAME, TaskStates, EMBEDDING_FIELD, MONGO_URL, MONGO_PORT, MONGO_USER, MONGO_PASS,
    MONGO_AUTH_DB, HS_CLUSTER_ADDRESS, GRPC_PROXY_ADDRESS,
)
from hydro_viz.utils import data_management


TransformResult = namedtuple('TransformResult', 'state raise_error meta result')


def get_embeddings(
        production_df: pd.DataFrame, 
        training_df: pd.DataFrame, 
        training_data_sample_size: int,
        hs_cluster: Cluster, 
        model: ModelVersion
):
    production_embeddings = data_management.parse_embeddings_from_dataframe(production_df)
    if training_df is None:
        training_embeddings = None
    else:
        try:
            servable_proto = Servable.create(cluster=hs_cluster, model_name=model.name, version=model.version)
            servable_proto.lock_while_starting()

            servable = Servable.find_by_name(hs_cluster, servable_name=servable_proto.name)
        except Exception as e:
            logging.error(f"Couldn't create {repr(model)} servable. Error:{e}")
            training_embeddings = None
        else:
            training_data_sample_size = min(training_data_sample_size, len(training_df))
            training_df_sample = training_df.sample(training_data_sample_size)
            training_embeddings = data_management.compute_training_embeddings(servable, training_df_sample)
            Servable.delete(hs_cluster, servable.name)
    return production_embeddings, training_embeddings


def generate_auto_embeddings(
        production_data: pd.DataFrame, 
        training_data: pd.DataFrame, 
        model: ModelVersion,
        encoder: Optional[AutoEmbeddingsEncoder]
) -> Tuple[Optional[List[np.array]], Optional[List[np.array]], Optional[AutoEmbeddingsEncoder]]:
    model_inputs = list(model.signature.inputs)
    input_names = [model_input.name for model_input in model_inputs]
    used_inputs = training_data.columns.intersection(input_names)
    training_feature_map = dataframe_to_feature_map(training_data[used_inputs], model)
    production_feature_map = dataframe_to_feature_map(production_data[used_inputs], model)

    if (training_feature_map is None or production_feature_map is None) or (
            len(training_feature_map) == 0 or len(production_feature_map) == 0):
        return None, None, None  # not enough data

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

    return [production_numerical_embeddings, production_categorical_embeddings], \
        [training_numerical_embeddings, training_categorical_embeddings], encoder


@celery_app.task(bind=True, track_started=True)
def transform_task(self, method, model_version_id):
    logging.info("start task")
    task_result: TransformResult = perform_transform_task(method, model_version_id)
    logging.info("counted")
    if task_result.raise_error:
        logging.error(task_result)
        logging.error(task_result.raise_error)
        logging.error(task_result.state)
        logging.error(task_result.meta)
        logging.error(task_result.result)
        self.update_state(state=task_result.state, meta=task_result.meta)
        raise Ignore()
    if task_result.result is not None:
        return task_result.result
    else:
        self.update_state(state=TaskStates.ERROR, meta={})
        raise Ignore()


def perform_transform_task(method: str, model_version_id: int) -> TransformResult:
    start = datetime.now()
    logging.info("start")
    s3_model_path = f's3://{HYDRO_VIS_BUCKET_NAME}/{model_version_id}'  # TODO make management of S3 bucket storage to check if model in storage is correct
    s3manager.fs.mkdirs(s3_model_path, exist_ok=True)

    hs_cluster = get_hs_cluster(HS_CLUSTER_ADDRESS, GRPC_PROXY_ADDRESS)
    mongo_client = get_mongo_client(MONGO_URL, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_AUTH_DB)
    mongo_collection = mongo_client["visualization"]

    db_model_info = data_management.get_record(mongo_collection, method, model_version_id)
    parameters = db_model_info.get('parameters', {})
    path_to_transformer = db_model_info.get('transformer_file', '')
    path_to_result_file = db_model_info.get('result_file', '')
    path_to_encoder = db_model_info.get('encoder_file', '')

    logging.info("read json")
    if path_to_result_file:
        plottable_result = s3manager.read_json(filepath=path_to_result_file)
        if plottable_result:
            return TransformResult(
                state=TaskStates.SUCCESS, 
                raise_error=False, 
                meta={}, 
                result={"result": plottable_result}
            )

    vis_metrics: List[str] = db_model_info.get(
        'visualization_metrics', DEFAULT_PROJECTION_PARAMETERS['visualization_metrics'])
    vis_metrics: List[VisMetrics] = [VisMetrics.to_enum(metric_name) for metric_name in vis_metrics]
    training_data_sample_size = db_model_info.get(
        'training_data_sample_size', DEFAULT_PROJECTION_PARAMETERS['training_data_sample_size'])
    production_data_sample_size = db_model_info.get(
        'production_data_sample_size', DEFAULT_PROJECTION_PARAMETERS['production_data_sample_size'])

    logging.info("try")
    try:
        model = ModelVersion.find_by_id(hs_cluster, int(model_version_id))
        model_name = model.name
        model_version = model.version
        embeddings_exist = data_management.model_has_correct_embeddings_field(model)
        if not embeddings_exist:
            message = f'Model {repr(model_version)} does not have {EMBEDDING_FIELD} field in output signature'
            logging.info(message)
            return TransformResult(
                state=TaskStates.NOT_SUPPORTED,
                raise_error=False,
                meta={'message': message, 'code': 400},
                result=None
            )
    except ValueError as e:
        return TransformResult(
            state=TaskStates.ERROR, 
            raise_error=True, 
            meta={'message': f"Error: {e}", 'code': 404},
            result=None
        )
    except Exception as e:
        return TransformResult(
            state=TaskStates.ERROR, 
            raise_error=True, 
            meta={'message': f"Error: {e}", 'code': 500},
            result=None
        )

    logging.info("joblib")
    transformer = s3manager.load_with_joblib(path_to_transformer) if path_to_transformer else None 
    autoembeddings_encoder = s3manager.load_with_joblib(path_to_encoder) if path_to_encoder else None

    # Parsing model requests and training data
    path_to_training_data = data_management.get_training_data_path(model)
    if path_to_training_data:
        try:
            training_df = pd.read_csv(s3manager.fs.open(path_to_training_data, mode='rb'))
        except:
            e = sys.exc_info()[0]
            logging.error(f'Couldn\'t get training data from {path_to_training_data}: {e}')
            training_df = None
    else:
        training_df = None
    
    logging.info("get prod subsample")
    production_requests_df = data_management.get_production_subsample(model.id, production_data_sample_size)
    if production_requests_df.empty:
        return TransformResult(
            state=TaskStates.NO_DATA, 
            raise_error=True,
            meta={'message': f'{repr(model)} model has not enough production data.', 'code': 404}, 
            result=None
        )
    if ((training_df is None) or training_df.empty) and (not embeddings_exist):
        return TransformResult(
            state=TaskStates.NO_DATA, 
            raise_error=True, 
            meta={'message': f'{repr(model)} model requires training data to generate autoembeddings.', 'code': 404},
            result=None
        )

    logging.info("ifexists")
    if embeddings_exist:
        production_embeddings, training_embeddings = get_embeddings(
            production_requests_df, training_df, training_data_sample_size, hs_cluster, model)
        logging.info("got embeddings")
        if production_embeddings is None or training_embeddings is None:
            if production_embeddings is None:
                message = "Production embeddings are unavailable"
            else:
                message = "Couldn't compute training embeddings"
            return TransformResult(
                state=TaskStates.NO_DATA,
                raise_error=True,
                meta={"message": message, "code": 500}, 
                result=None
            )
    else:
        logging.info("if not exists")
        [production_numerical_embeddings, production_categorical_embeddings], \
            [training_numerical_embeddings, training_categorical_embeddings], \
            autoembeddings_encoder = generate_auto_embeddings(
                production_requests_df, training_df, model, encoder=autoembeddings_encoder)
        logging.info("generated auto emb")
        if training_numerical_embeddings is not None and training_categorical_embeddings is None:  # Only numerical features are used
            production_embeddings, training_embeddings = production_numerical_embeddings, training_numerical_embeddings
        elif training_categorical_embeddings is not None and training_numerical_embeddings is None:  # Only categorical features are used
            production_embeddings, training_embeddings = production_categorical_embeddings, training_categorical_embeddings
            parameters['metric'] = 'dice'
        elif training_categorical_embeddings is not None and training_numerical_embeddings is not None:  # Both
            production_embeddings, training_embeddings = \
                [production_numerical_embeddings, production_categorical_embeddings], \
                [training_numerical_embeddings, training_categorical_embeddings]
        else:
            return TransformResult(
                state=TaskStates.NOT_SUPPORTED, 
                raise_error=True, 
                meta={'message': f"Couldn\'t extract autoembeddings from {model}.", 'code': 404},
                result=None
            )

    if isinstance(training_embeddings, np.ndarray):  
        # not mixed-type data
        logging.info("calculate")
        top_N_neighbours = data_management.calcualte_neighbours(production_embeddings)
        plottable_result, transformer = transform_high_dimensional(
            method, parameters, training_embeddings, production_embeddings, transformer, vis_metrics=vis_metrics)
    else:  
        # mixed-type data
        top_N_neighbours = []
        plottable_result, transformer = transform_high_dimensional_mixed(
            method, parameters, training_embeddings, production_embeddings)

    logging.info("got plottable res")
    requests_data_dict = data_management.parse_requests_dataframe(
        production_requests_df, hs_cluster, model, top_N_neighbours)
    plottable_result.update(requests_data_dict)

    logging.info("parsed request data frame")

    path_to_result_file = s3_model_path + '/result.json'
    try:
        s3manager.write_json(data=plottable_result, filepath=path_to_result_file)
        db_model_info["result_file"] = path_to_result_file
    except:
        e = sys.exc_info()[1]
        logging.error(f'Couldn\'t save result to {path_to_result_file}: {e}')

    logging.info("wrote to s3")

    if type(transformer) != UmapTransformerWithMixedTypes:  
        # MixedTypes is used only with fit_transform
        transformer_path = s3_model_path + f'/transformer_{method}_{model_name}{model_version}'
        try:
            transformer_saved_to_s3 = s3manager.dump_with_joblib(transformer, filepath=transformer_path)
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

    data_management.update_record(mongo_collection, method, db_model_info, model.id)
    logging.info(f'Request handled in {datetime.now() - start}')
    return TransformResult(
        state=TaskStates.SUCCESS, 
        raise_error=False, 
        meta={}, 
        result={"result": plottable_result}
    )
