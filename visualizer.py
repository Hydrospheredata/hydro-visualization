from data_management import S3Manager, get_requests_data
from ml_transformers.transformer import UmapTransformer
from ml_transformers.utils import get_top_100
from loguru import logger
import numpy as np
from datetime import datetime
import pandas as pd


def visualize_high_dimensional(model_name, model_version, method, bucket, request_files=[], profile_file=[],
                               vis_metrics=["global_score", "sammon_error", "auc_score", "stability_score", "msid", "clustering"]):
    """
    Visualizes high dimensional data
    TODO add profile management, add outlier labels management, etc.
    :param model_name: name of a model
    :param model_version: version
    :param method: 'umap'
    :param request_files: list of S3 files with requests
    :param profile_file: list of S3 files of data profile
    :return: json with points
    """
    result = {}
    start = datetime.now()
    embeddings, predictions, confidences, \
    anomaly_predictions, anomaly_confidences = np.empty(0), np.empty(0), np.empty(0), \
                                               np.empty(0), np.empty(0)
    parameters = get_transformer_parameters(model_name, model_version, bucket)
    ml_transformer = None
    if method == 'umap':
        ml_transformer = UmapTransformer(parameters)
    if ml_transformer is None:
        logger.error('Cannot define transformer. Illegal method name')
    if request_files:
        embeddings, predictions, confidences, \
        anomaly_predictions, anomaly_confidences = get_requests_data(bucket, request_files)
        logger.info(f'Parsing data took {datetime.now() - start}')
    if len(embeddings) == 0:
        logger.error(f'Could not get embeddiings from bucket {bucket}, {request_files}')
        return {}

    start = datetime.now()
    transformed_embeddings = ml_transformer.fit_transform(
        embeddings)  # TODO add ground truth labels management for semi-supervised umap
    logger.info(f'Fitting {embeddings.shape[0]} {embeddings.shape[1]}-dimensional points took {datetime.now() - start}')

    vis_eval_metrics = ml_transformer.eval(embeddings, transformed_embeddings, y=None, evaluation_metrics=vis_metrics) # TODO add ground truth_labels
    top_100_neighbours = get_top_100(embeddings)
    result['data_shape'] = transformed_embeddings.shape
    result['data'] = pd.Series(transformed_embeddings.flatten()).to_json(orient='values')
    result['class_labels'] = {'ground_truth': [],
                              'predicted': predictions.tolist(),
                              'confidences': confidences.tolist()}
    result['metrics'] = {'anomality': {'scores':anomaly_confidences.tolist(),
                                       'threshold': 0.5}
                         }
    result['top_100'] = top_100_neighbours
    result['visualization_metrics'] = vis_eval_metrics
    return result


def get_transformer_parameters(model_name, model_version, bucket):
    """
    Searchers for parameters json on bucket, if found uses new parameters, else uses default
    :param model_name:
    :param model_version:
    :param bucket:
    :return: dict with parameters or empty
    """

    return {}
