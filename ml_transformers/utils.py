from datetime import datetime
from enum import Enum
from typing import List

import numpy as np
from loguru import logger as logging
from scipy.spatial import cKDTree


class Coloring(Enum):
    CLASS = 'class'
    GRADIENT = 'gradient'
    NONE = 'none'


class VisMetrics(Enum):
    global_score = 'global_score',
    sammon_error = 'sammon_error',
    auc_score = 'auc_score',
    stability = 'stability',
    msid = 'msid',
    clustering = 'clustering'

    @classmethod
    def has_key(cls, name):
        return any(x for x in cls if x.name == name)

    @classmethod
    def has_val(cls, value):
        return any(x for x in cls if value in x.value)

    @classmethod
    def to_enum(cls, name):
        for val in cls:
            if val.name == name:
                return val
        return None


AVAILBALE_VIS_METRICS = list(VisMetrics)

AVAILABLE_TRANSFORMERS = {'umap'}  # {'umap', 'tsne', 'trimap'}
DEFAULT_TRANSFORMER_PARAMETERS = {'umap':
                                      {'min_dist': 0.1, 'n_neighbours': 15, 'metric': 'euclidean', 'n_components': 2}
                                  }

DEFAULT_PROJECTION_PARAMETERS = {'parameters': DEFAULT_TRANSFORMER_PARAMETERS,
                                 'use_labels': False,
                                 'visualization_metrics': [VisMetrics.global_score.name],
                                 'training_data_sample_size': 5000,
                                 'production_data_sample_size': 500}


def get_top_N_neighbours(X, N=50) -> List[List[int]]:
    """
    Finds top 100 closest neighbours for each point
    :param X: list of points in high dimensional space
    :return: 2D list of 100 neighbours indices for each point
    """
    start = datetime.now()
    tree = cKDTree(X)
    top_100 = []
    for i in range(len(X)):
        _, top = tree.query(X[i], k=N + 1)
        top = np.delete(top, np.where(top == i))
        top_100.append(top.tolist())
    logging.info(f'TOP 100 neighbour calculation took: {datetime.now() - start}')
    return top_100
