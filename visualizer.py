from datetime import datetime

import numpy as np
from loguru import logger

from ml_transformers.transformer import UmapTransformer
from ml_transformers.utils import get_top_100


def visualize_high_dimensional(method, parameters, training_embeddings, requests_embeddings, transformer_instance=None,
                               vis_metrics=["global_score", "sammon_error", "auc_score", "stability_score", "msid",
                                            "clustering"]):
    """
    Visualizes high dimensional data
    :param model_name: name of a model
    :param model_version: version
    :param method: 'umap'
    :param request_files: list of S3 files with requests
    :param profile_file: list of S3 files of data profile
    :return: json with points if requests_embeddings are present, {}, None otherwise
    """
    result = {}
    ml_transformer = None
    need_fit = True

    if method == 'umap':
        if transformer_instance and isinstance(transformer_instance, UmapTransformer):
            need_fit = False
            ml_transformer = transformer_instance
        else:
            ml_transformer = UmapTransformer(parameters)
    if ml_transformer is None:
        logger.error('Cannot define transformer. Illegal method name')

    if training_embeddings is not None and requests_embeddings is not None:
        total_embeddings = np.concatenate([requests_embeddings, training_embeddings])
    elif requests_embeddings is not None:
        total_embeddings = requests_embeddings
    else:
        return {}, None

    start = datetime.now()
    if not need_fit and method == 'umap':
        transformed_embeddings = ml_transformer.transform(requests_embeddings)
    else:
        transformed_embeddings = ml_transformer.fit_transform(
            total_embeddings)  # TODO add ground truth labels management for semi-supervised umap

    logger.info(
        f'Fitting {total_embeddings.shape[0]} {total_embeddings.shape[1]}-dimensional points took {datetime.now() - start}')

    vis_eval_metrics = ml_transformer.eval(total_embeddings, transformed_embeddings, y=None,
                                           evaluation_metrics=vis_metrics)  # TODO add ground truth_labels
    top_100_neighbours = get_top_100(requests_embeddings)
    transformed_embeddings = transformed_embeddings[:len(requests_embeddings)]
    result['data_shape'] = transformed_embeddings.shape
    result['data'] = transformed_embeddings.tolist()
    result['top_100'] = top_100_neighbours
    result['visualization_metrics'] = vis_eval_metrics
    ml_transformer.embedding_ = None
    return result, ml_transformer

