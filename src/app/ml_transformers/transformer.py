import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import umap
import logging
from umap import UMAP

from app.ml_transformers.metrics import (
    global_score, sammon_error, stability_score, auc_score, 
    intristic_multiscale_score, clustering_score
)
from app.ml_transformers.utils import (
    DEFAULT_TRANSFORMER_PARAMETERS, VisMetrics, AVAILBALE_VIS_METRICS
)


class Transformer(ABC):
    def __init__(self, parameters: Dict):
        self._set_params(parameters)
        self.__instance__, self.transformer = self._create()  # clear instance of transformer (not fitted into data)

    @abstractmethod
    def _set_params(self, parameters: Dict):
        pass

    @abstractmethod
    def _create(self):
        return None, None

    @abstractmethod
    def fit(self, X: np.ndarray, y=None):
        pass

    @abstractmethod
    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return None

    @abstractmethod
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return None

    def eval(self, X: np.ndarray, _X: np.ndarray, y=None,
             evaluation_metrics: Tuple[VisMetrics] = tuple(AVAILBALE_VIS_METRICS),
             _auc_cv=5) -> Dict[str, str]:
        """
        Evaluates vizualization using listed evaluation_metrics names
        :param X: original points
        :param _X: transformed points
        :param y: labels
        :param evaluation_metrics: list of metrics names
        :param _auc_cv: number of splits for acc evaluation
        :return: dict of metric values
        """
        start = datetime.now()
        eval_metrics = {}
        if VisMetrics.GLOBAL_SCORE in evaluation_metrics:
            eval_metrics[VisMetrics.GLOBAL_SCORE.value] = str(global_score(X, _X))
        if VisMetrics.SAMMON_ERROR in evaluation_metrics:
            eval_metrics[VisMetrics.SAMMON_ERROR.value] = str(sammon_error(X, _X))
        if VisMetrics.AUC_SCORE in evaluation_metrics and y is not None:
            acc_result = auc_score(_X, y)
            eval_metrics['knn_acc'] = str(acc_result['knn_acc'])  # TODO knn_acc
            eval_metrics['svc_acc'] = str(acc_result['svc_acc'])
        if VisMetrics.STABILITY in evaluation_metrics:
            eval_metrics[VisMetrics.STABILITY.value] = str(stability_score(X, self.__instance__))
        if VisMetrics.MSID in evaluation_metrics:
            eval_metrics[VisMetrics.MSID.value] = str(intristic_multiscale_score(X, _X))
        if VisMetrics.CLUSTERING in evaluation_metrics and y is not None:
            ars, ami = clustering_score(_X, y)
            eval_metrics['clustering_random_score'] = str(ars)  # TODO svc_acc
            eval_metrics['clustering_mutual_info'] = str(ami)
        logging.info(f'Evaluation of embeddings took {datetime.now() - start}')
        return eval_metrics


class UmapTransformer(Transformer):
    def __init__(self, parameters):
        self.default_parameters = DEFAULT_TRANSFORMER_PARAMETERS['umap']
        super().__init__(parameters)
        self._embedding_ = None

    def _set_params(self, parameters):
        self.min_dist = parameters.get('min_dist', self.default_parameters['min_dist'])
        self.n_neighbours = parameters.get('n_neighbours', self.default_parameters['n_neighbours'])
        self.metric = parameters.get('metric', self.default_parameters['metric'])
        self.n_components = parameters.get('n_components', self.default_parameters['n_components'])
        self.use_labels = parameters.get('use_labels', False)

    def _create(self):
        transformer = UMAP(n_neighbors=self.n_neighbours,
                           n_components=self.n_components,
                           min_dist=self.min_dist,
                           metric=self.metric)
        __instance__ = UMAP(n_neighbors=self.n_neighbours,  # unfitted instance with same parameters for
                            n_components=self.n_components,  # stability metric
                            min_dist=self.min_dist,
                            metric=self.metric)
        return transformer, __instance__

    def fit(self, X, y=None):
        warnings.filterwarnings('ignore')
        if y is None:
            self.transformer.fit(X)
        elif self.use_labels:
            self.transformer.fit(X, y)
        else:
            self.transformer.fit(X)
        self._embedding_ = self.transformer.embedding_

    def fit_transform(self, X, y=None):
        warnings.filterwarnings('ignore')
        if self.use_labels:
            _X = self.transformer.fit_transform(X, y)
        else:
            _X = self.transformer.fit_transform(X)
        self._embedding_ = _X
        return _X

    def transform(self, X, y=None):
        warnings.filterwarnings('ignore')
        if self.use_labels or y is not None:
            _X = self.transformer.transform(X, y)
        else:
            _X = self.transformer.transform(X)
        return _X


class UmapTransformerWithMixedTypes(UmapTransformer):
    """
    Special instance of umap transformer with handling of two transformers for numerical and categorical data.
    """

    def __init__(self, parameters):
        super().__init__(parameters)
        self.default_parameters = DEFAULT_TRANSFORMER_PARAMETERS['umap_mixed']
        self.categorical_weight = parameters.get('categorical_weight', self.default_parameters['categorical_weight'])
        (self.transformer, self.transformer_categorical), \
            (self.__instance__, self.__instance_categorical) = self._create()
        self.numerical_embedding_ = None
        self.embedding_ = None
        self.categorical_embedding_ = None

    def _create(self):
        """
        Creates separate instances for numerical and categorical types
        :return:
        """
        transformer = UMAP(
            n_neighbors=self.n_neighbours,
            n_components=self.n_components,
            min_dist=self.min_dist,
            metric=self.metric
        )
        transformer_categorical = UMAP(
            n_neighbors=self.n_neighbours,
            n_components=self.n_components,
            min_dist=self.min_dist,
            metric='jaccard'
        )
        __instance__ = UMAP(
            n_neighbors=self.n_neighbours,  # unfitted instance with same parameters for
            n_components=self.n_components,  # stability metric
            min_dist=self.min_dist,
            metric=self.metric
        )
        __instance_categorical__ = UMAP(
            n_neighbors=self.n_neighbours,  # unfitted instance with same parameters for
            n_components=self.n_components,  # stability metric
            min_dist=self.min_dist,
            metric='jaccard'
        )
        return (transformer, transformer_categorical), (__instance__, __instance_categorical__)

    def fit(self, X, X_categorical=None, y=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.fit_transform(X, X_categorical)

    def fit_transform(self, X, X_categorical=None, y=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if X_categorical is None:
                raise ValueError('X_categorical should should not be None')
            elif X is None:
                raise NotImplementedError('This transformer do not support fitting of only categorical data')
            else:
                if self.use_labels:
                    self.transformer.fit(X, y)
                    self.transformer_categorical.fit(X_categorical, y)
                else:
                    self.transformer.fit(X)
                    self.transformer_categorical.fit(X_categorical)
                
                intersection = umap.umap_.general_simplicial_set_intersection(
                    self.transformer_categorical.graph_,
                    self.transformer.graph_,
                    weight=self.categorical_weight
                )
                intersection = umap.umap_.reset_local_connectivity(intersection)
                embedding = umap.umap_.simplicial_set_embedding(
                    self.transformer_categorical._raw_data,
                    intersection, self.transformer_categorical.n_components,
                    self.transformer_categorical._initial_alpha,
                    self.transformer_categorical._a,
                    self.transformer_categorical._b,
                    self.transformer_categorical.repulsion_strength,
                    self.transformer_categorical.negative_sample_rate,
                    200, 'random', np.random,
                    self.transformer_categorical.metric,
                    self.transformer_categorical._metric_kwds, False
                )

                self.numerical_embedding_ = self.transformer.embedding_
                self.categorical_embedding_ = self.transformer_categorical.embedding_
                self.embedding_ = embedding
        return embedding

    def transform(self, X, X_categorical=None, y=None):
        """
        Is not implemented since UMAP doesn't have this functionality for graph intersetions
        """
        raise NotImplementedError(f'{self.__class__.__name__} cannot transform data. Use fit_transform() instead.')

    def eval(
            self, 
            X: np.ndarray, 
            _X: np.ndarray, 
            y=None,
            evaluation_metrics: Tuple[VisMetrics] = tuple(AVAILBALE_VIS_METRICS),
            _auc_cv=5
    ) -> Dict[str, str]:
        """
        Is not implemented since some metrics require complex distance function for mixed-type
        """
        raise NotImplementedError(f'{self.__class__.__name__} doesn\'t implement evaluation of transformation.')


def transform_high_dimensional(
        method: str, 
        parameters: Dict,
        training_embeddings: Optional[np.ndarray],
        production_embeddings: np.ndarray,
        transformer_instance: Optional[Transformer] = None,
        vis_metrics: Tuple[VisMetrics] = tuple(AVAILBALE_VIS_METRICS)
) -> Tuple[Dict, Transformer]:
    """
    Transforms data of one type (only numerical or only categorical) from higher dimensions to lower.
     And calculates vis_metrics of transformation.
    :param method: transformer method
    :param parameters: {}
    :param training_embeddings: 2D numpy array
    :param production_embeddings: 2D numpy array
    :param transformer_instance: Transformer or None. If None then Transformer is created and fit_transform() method is used.
    If is not None then transform() is used
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
        logging.error('Cannot define transformer. Illegal method name')

    if training_embeddings is not None and production_embeddings is not None:
        total_embeddings = np.concatenate([production_embeddings, training_embeddings])
    else:
        total_embeddings = production_embeddings  # 100

    start = datetime.now()
    if not need_fit and method == 'umap':
        plottable_embeddings = transformer.transform(total_embeddings)
    else:
        plottable_embeddings = transformer.fit_transform(
            total_embeddings)  # TODO add ground truth labels management for semi-supervised umap
    logging.info(
        f'Fitting {total_embeddings.shape[0]} {total_embeddings.shape[1]}-dimensional points took '
        f'{datetime.now() - start}')

    vis_eval_metrics = transformer.eval(total_embeddings, plottable_embeddings,
                                        y=None,
                                        evaluation_metrics=vis_metrics)  # TODO add ground truth_labels

    plottable_training_embeddings = plottable_embeddings[len(production_embeddings):]
    plottable_prod_embeddings = plottable_embeddings[:len(production_embeddings)]
    result['data_shape'] = plottable_prod_embeddings.shape
    result['data'] = plottable_prod_embeddings.tolist()
    result['visualization_metrics'] = vis_eval_metrics
    if len(plottable_training_embeddings) > 0:
        result['training_data'] = plottable_training_embeddings.tolist()
        result['training_data_shape'] = plottable_training_embeddings.shape
    transformer._embedding_ = None  # ?

    return result, transformer


def transform_high_dimensional_mixed(
        method: str, 
        parameters: Dict,
        training_embeddings: List[np.ndarray],
        production_embeddings: List[np.ndarray]
) -> Tuple[Dict, Transformer]:
    """
    Transforms data of mixed type. This method uses only fit_transform method and requires 
    specific transformer that handlex mixed types. It doesn't compute vis_metrics of transformation.
    
    :param method: transformer method
    :param parameters: Dict
    :param training_embeddings: List[np.array] has two numpy arrays - [numerical_embeddings, categorical_embeddings]
    :param production_embeddings: List[np.array] has two numpy arrays - [numerical_embeddings, categorical_embeddings]
    :return: (embedding dict, transformer)
    """
    result = {}

    if method == 'umap':
        transformer = UmapTransformerWithMixedTypes(parameters)
    else:
        logging.error('Cannot define transformer. Illegal method name')

    total_numerical_embeddings = np.concatenate([production_embeddings[0], training_embeddings[0]])
    total_categorical_embeddings = np.concatenate([production_embeddings[1], training_embeddings[1]])
    plottable_embeddings = transformer.fit_transform(X=total_numerical_embeddings,
                                                     X_categorical=total_categorical_embeddings)

    plottable_training_embeddings = plottable_embeddings[len(production_embeddings[0]):]
    plottable_prod_embeddings = plottable_embeddings[:len(production_embeddings[0])]
    result['data_shape'] = plottable_prod_embeddings.shape
    result['data'] = plottable_prod_embeddings.tolist()
    if len(plottable_training_embeddings) > 0:
        result['training_data'] = plottable_training_embeddings.tolist()
        result['training_data_shape'] = plottable_training_embeddings.shape

    return result, transformer
