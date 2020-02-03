from datetime import datetime
from typing import List

import numpy as np
from loguru import logger
from scipy.spatial import cKDTree

AVAILABLE_TRANSFORMERS = {'umap'}  # {'umap', 'tsne', 'trimap'}

UMAP_N_NEIGHBOURS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = 'euclidean'

DEFAULT_UMAP_PARAMETERS = {'min_dist': 0.1, 'n_neighbours': 15, 'metric': 'euclidean', 'n_components': 2}


def get_top_100(X) -> List[List[int]]:
    '''
    Finds top 100 closest neighbours for each point
    :param X: list of points in high dimensional space
    :return: 2D list of 100 neighbours indices for each point
    '''
    start = datetime.now()
    tree = cKDTree(X)
    top_100 = []
    for i in range(len(X)):
        _, top = tree.query(X[i], k=101)
        top = np.delete(top, np.where(top == i))
        top_100.append(top.tolist())
    logger.info(f'TOP 100 nighbour calculation took: {datetime.now() - start}')
    return top_100
