from datetime import datetime
from typing import Optional

import numpy as np
from loguru import logger

from ml_transformers.transformer import UmapTransformer, Transformer


def transform_high_dimensional(method, parameters,
                               training_embeddings: Optional[np.ndarray],
                               production_embeddings: np.ndarray,
                               transformer_instance: Optional[Transformer] = None,
                               vis_metrics=["global_score", "sammon_error", "auc_score", "stability_score", "msid",
                                            "clustering"]):
    """
    Transforms data from higher dimensions to lower
    :param method: transformer method
    :param parameters: {}
    :param training_embeddings:
    :param production_embeddings:
    :param transformer_instance:
    :param vis_metrics:
    :return: (embedding dict, transformer)
    """
    result = {}
    transformer = None
    need_fit = True

    if method == 'umap':
        if transformer_instance and isinstance(transformer_instance, UmapTransformer):
            need_fit = False
            transformer = transformer_instance
        else:
            transformer = UmapTransformer(parameters)
    if transformer is None:
        logger.error('Cannot define transformer. Illegal method name')

    if training_embeddings is not None and production_embeddings is not None:
        total_embeddings = np.concatenate([production_embeddings, training_embeddings])
    else:
        total_embeddings = production_embeddings

    start = datetime.now()
    if not need_fit and method == 'umap':
        plottable_embeddings = transformer.transform(production_embeddings)
    else:
        plottable_embeddings = transformer.fit_transform(
            total_embeddings)  # TODO add ground truth labels management for semi-supervised umap

    logger.info(
        f'Fitting {total_embeddings.shape[0]} {total_embeddings.shape[1]}-dimensional points took '
        f'{datetime.now() - start}')

    vis_eval_metrics = transformer.eval(total_embeddings[:len(plottable_embeddings)], plottable_embeddings,
                                        y=None,
                                        evaluation_metrics=vis_metrics)  # TODO add ground truth_labels

    plottable_embeddings = plottable_embeddings[:len(production_embeddings)]  # Slice excessive ?

    result['data_shape'] = plottable_embeddings.shape
    result['data'] = plottable_embeddings.tolist()
    result['visualization_metrics'] = vis_eval_metrics

    transformer.embedding_ = None  # ?

    return result, transformer
