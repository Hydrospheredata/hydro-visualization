import json
import logging
import random
import tempfile
import time
from typing import Dict, Optional, List, Any, Iterable

import joblib
import numpy as np
import pandas as pd
import requests

from hydrosdk.cluster import Cluster
from hydrosdk.signature import ModelField, ProfilingType
from hydrosdk.modelversion import ModelVersion
from hydrosdk.monitoring import MetricSpec
from hydrosdk.servable import Servable

from hydro_viz.ml_transformers.utils import DEFAULT_TRANSFORMER_PARAMETERS, Coloring, get_top_N_neighbours, \
    DEFAULT_PROJECTION_PARAMETERS
from hydro_viz.utils.conf import AWS_STORAGE_ENDPOINT, AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, \
    HS_CLUSTER_ADDRESS, HYDRO_VIS_BUCKET_NAME, EMBEDDING_FIELD, N_NEIGHBOURS
from hydro_viz.ml_transformers.autoembeddings import NOT_IGNORED_PROFILE_TYPES


def calcualte_neighbours(embeddings: np.array) -> List[List[int]]:
    n_neighbours = min(N_NEIGHBOURS + 1, len(embeddings) - 1)
    top_n_neighbours = get_top_N_neighbours(embeddings, N=n_neighbours)
    return top_n_neighbours


def parse_requests_dataframe(
            df: pd.DataFrame, 
            hs_cluster: Cluster, 
            model: ModelVersion,
            top_n_neighbours: List[List[int]]
) -> Dict:
    """
    Extracts:
        - model scalar outputs values
        - all model monitoring metrics and thresholds
    from requests dataframe
    :param df: dataframe
    :param monitoring_fields: list of monitoring metrics names with comparison operator
    :return: Dict {"output_info":{…}, "metrics":{…}}
    """

    def get_coloring_info(model_field: ModelField) -> Coloring:
        if ProfilingType(model_field.profile) in [ProfilingType.CATEGORICAL, ProfilingType.NOMINAL,
                                                  ProfilingType.ORDINAL]:
            return Coloring.CLASS
        elif ProfilingType(model_field.profile) in [ProfilingType.NUMERICAL, ProfilingType.CONTINUOUS,
                                                    ProfilingType.INTERVAL, ProfilingType.RATIO]:
            return Coloring.GRADIENT
        else:
            return Coloring.NONE

    monitoring_fields = [
        (metric.name, metric.config.threshold_op, metric.config.threshold) 
        for metric in MetricSpec.find_by_modelversion(hs_cluster, model.id)
    ]
    scalar_model_outputs: List[ModelField] = list(
        filter(lambda x: len(x.shape.dims) == 0,
        model.signature.outputs
    ))
    requests_ids = df['_id'].values.tolist()

    output_info = {}
    counterfactuals = [[] for _ in range(len(top_n_neighbours))]
    for scalar_output in scalar_model_outputs:
        coloring_type = get_coloring_info(scalar_output)
        field_info = {
            'data': df[scalar_output.name].values.tolist(), 
            'coloring_type': coloring_type.value,
            'dtype': scalar_output.dtype
        }
        if coloring_type == Coloring.CLASS:
            field_info['classes'] = np.unique(df[scalar_output.name].values).tolist()
        if scalar_output.name == 'class':  # fixme add counterfactuals by name
            class_labels = df[scalar_output.name].values.tolist()
            if len(top_n_neighbours) > 0:
                counterfactuals = list(map(
                    lambda i: list(filter(lambda x: class_labels[x] != class_labels[i], top_n_neighbours[i])),
                    range(len(top_n_neighbours))
                ))
            del class_labels
        output_info[scalar_output.name] = field_info

    monitoring_data = {}
    metric_checks = df._hs_metric_checks.to_list()
    for monitoring_metric_name, comparison_operator, threshold in monitoring_fields:
        monitoring_data[monitoring_metric_name] = {
            'scores': [], 
            'threshold': threshold,
            'operation': comparison_operator
        }
    for request in metric_checks:
        for monitoring_metric_name, comparison_operator, threshold in monitoring_fields:
            metric_dict = request.get(monitoring_metric_name, {})
            metric_data = monitoring_data[monitoring_metric_name]
            metric_data['scores'].append(metric_dict.get('value', None))
            monitoring_data[monitoring_metric_name] = metric_data
    for monitoring_metric_name, comparison_operator, threshold in monitoring_fields:
        monitoring_data[monitoring_metric_name].update({'coloring_type': Coloring.GRADIENT.value})

    return {
        'output_info': output_info,
        'metrics': monitoring_data,
        'requests_ids': requests_ids,
        'top_N': top_n_neighbours,
        'counterfactuals': counterfactuals
    }


def parse_embeddings_from_dataframe(df: pd.DataFrame):
    return np.apply_along_axis(
        lambda x: np.array(x), arr=df[EMBEDDING_FIELD].values.tolist(), axis=0)


def get_record(db, method: str, model_version_id: [str, int]) -> Dict:
    model_version_id = str(model_version_id)
    existing_record = db[method].find_one({"model_version_id": model_version_id})
    if not existing_record:
        return {
            "model_version_id": model_version_id,
            "result_file": "",
            "transformer_file": "",
            "parameters": DEFAULT_TRANSFORMER_PARAMETERS[method],
            "visualization_metrics": DEFAULT_PROJECTION_PARAMETERS['visualization_metrics'],
            "production_data_sample_size": DEFAULT_PROJECTION_PARAMETERS['production_data_sample_size'],
            "training_data_sample_size": DEFAULT_PROJECTION_PARAMETERS['training_data_sample_size']
        }
    else:
        return existing_record


def update_record(db, method: str, record: dict, model_version_id: [int, str]):
    model_version_id = str(model_version_id)
    if '_id' in record:
        del record['_id']
    db[method].update_one(
        {"model_version_id": model_version_id},
        {"$set": record}, upsert=True)


def predict_with_circuit_breaker(predictor, request, retries=3, timeout=5):
    try:
        return predictor.predict(request)
    except Exception as e:
        if retries == 0:
            raise e
        else:
            time.sleep(timeout)
            return predict_with_circuit_breaker(predictor, request, retries-1, timeout*2)


def compute_training_embeddings(
        servable: Servable, 
        training_data: pd.DataFrame
) -> Optional[np.ndarray]:
    """
    Computes embeddings from training data using unmonitorable servable
    :param model: model instance
    :param servable: servable
    :param training_data: model training data dataframe
    :return: np.array [N, embedding_dim]
    """
    predictor = servable.predictor(monitorable=False)
    embeddings = []
    model_inputs_names = [i.name for i in predictor.signature.inputs]
    n_samples = len(training_data)
    logging.info("Inferencing training embeddings.")
    for i in range(n_samples):
        try:
            sample = training_data.iloc[i].to_dict()
            request = {k: v for k, v in sample.items() if k in model_inputs_names}
            result = predict_with_circuit_breaker(predictor, request)
        except Exception as e:
            logging.error(f'Couldn\'t get prediction of a data sample: {e}')
            return None
        embeddings.append(result[EMBEDDING_FIELD])
    embeddings = np.concatenate(embeddings, axis=0)
    logging.info(f'Inferenced training embeddings, shape: {embeddings.shape}')
    return embeddings


def get_production_subsample(model_id, size=1000) -> pd.DataFrame:
    r = requests.get(f'{HS_CLUSTER_ADDRESS}/monitoring/checks/subsample/{model_id}?size={size}')
    if r.status_code != 200:
        return pd.DataFrame()

    checks = r.json()
    checksWithoutError = list(filter(lambda x: x.get('_hs_error') == None, checks))
    return pd.DataFrame.from_dict(checksWithoutError)


def model_has_production_data(model_id) -> bool:
    production_data_aggregates = requests.get(
        f"{HS_CLUSTER_ADDRESS}/monitoring/checks/aggregates/{model_id}",
        params={"limit": 1, "offset": 0}).json()
    number_of_production_requests = production_data_aggregates['count']
    return number_of_production_requests > 0


def model_has_correct_embeddings_field(model: ModelVersion) -> bool:
    """
    Check if model returns EMBEDDING_FIELD with shape of vector [1, N]
    :param model:
    :return: True if returns EMBEDDING_FIELD with shape of vector [1, N]
    """
    output_names = { field.name : field for field in model.signature.outputs }
    if EMBEDDING_FIELD not in output_names:
        logging.info(f'Model {model.name}v{model.version} has no output {EMBEDDING_FIELD} field')
        return False
    embedding_field_shape = list(output_names[EMBEDDING_FIELD].shape.dims)
    shape_is_correct = len(embedding_field_shape) == 2 and embedding_field_shape[0] == 1
    logging.info(f'Model {repr(model)}. {EMBEDDING_FIELD} field shape is '
                 f'{"OK." if shape_is_correct else f"not OK. Expected: [1, any], Got:{embedding_field_shape}. "}')
    return shape_is_correct


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


def get_scalar_input_fields_with_profile(mv: ModelVersion) -> Iterable[ModelField]:
    for field in mv.signature.inputs:
        p1 = ProfilingType(field.profile) in NOT_IGNORED_PROFILE_TYPES
        p2 = len(field.shape.dims) == 0
        if p1 and p2:
            yield field
