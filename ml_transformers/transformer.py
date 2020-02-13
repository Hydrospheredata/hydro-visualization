import warnings
from abc import ABC, abstractmethod
from datetime import datetime

from loguru import logger
from umap import UMAP

from .metrics import global_score, sammon_error, stability_score, auc_score, intristic_multiscale_score, \
    clustering_score
from .utils import DEFAULT_PARAMETERS


class Transformer(ABC):
    def __init__(self, parameters):
        self.__set_params__(parameters)
        self.__instance__, self.transformer = self.__create__()  # clear instance of transformer (not fitted into data)

    @abstractmethod
    def __set_params__(self, parameters):
        pass

    @abstractmethod
    def __create__(self):
        return None, None

    @abstractmethod
    def fit(self, X, y=None):
        return None, None

    @abstractmethod
    def transform(self, X, y=None):
        return None, None

    @abstractmethod
    def fit_transform(self, X, y=None):
        return None

    def eval(self, X, _X, y=None,
             evaluation_metrics=["global_score", "sammon_error",
                                 "auc_score", "stability_score", "msid", "clustering"],
             _auc_cv=5):
        '''
        Evaluates vizualization using listed evaluation_metrics names
        :param X: original points
        :param _X: transformed points
        :param y: labels
        :param evaluation_metrics: list of metrics names
        :param _auc_cv: number of splits for acc evaluation
        :return: dict of metric values
        '''
        start = datetime.now()
        eval_metrics = {}
        if 'global_score' in evaluation_metrics:
            eval_metrics['global_score'] = str(global_score(X, _X))
        if 'sammon_error' in evaluation_metrics:
            eval_metrics['sammon_error'] = str(sammon_error(X, _X))
        if 'auc_score' in evaluation_metrics and y is not None:
            acc_result = auc_score(_X, y)
            eval_metrics['knn_acc'] = str(acc_result['knn_acc'])
            eval_metrics['svc_acc'] = str(acc_result['svc_acc'])
        if 'stability_score' in evaluation_metrics:
            eval_metrics['stability'] = str(stability_score(X, self.__instance__))
        if 'msid' in evaluation_metrics:
            eval_metrics['msid'] = str(intristic_multiscale_score(X, _X))
        if 'clustering' in evaluation_metrics and y is not None:
            ars, ami = clustering_score(_X, y)
            eval_metrics['clustering_random_score'] = str(ars)
            eval_metrics['clustering_mutual_info'] = str(ami)
        logger.info(f'Evaluation of embeddings took {datetime.now() - start}')
        return eval_metrics


class UmapTransformer(Transformer):
    def __init__(self, parameters):
        self.default_parameters = DEFAULT_PARAMETERS['umap']
        super().__init__(parameters)
        self.embedding_ = None

    def __set_params__(self, parameters):
        self.min_dist = parameters.get('min_dist', self.default_parameters['min_dist'])
        self.n_neighbours = parameters.get('n_neighbours', self.default_parameters['n_neighbours'])
        self.metric = parameters.get('metric', self.default_parameters['metric'])
        self.n_components = parameters.get('n_components', self.default_parameters['n_components'])
        self.use_labels = parameters.get('use_labels', False)

    def __create__(self):
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
        self.embedding_ = self.transformer.embedding_

    def fit_transform(self, X, y=None):
        warnings.filterwarnings('ignore')
        if self.use_labels:
            _X = self.transformer.fit_transform(X, y)
        else:
            _X = self.transformer.fit_transform(X)
        self.embedding_ = _X
        return _X

    def transform(self, X, y=None):
        warnings.filterwarnings('ignore')
        if self.use_labels or y is not None:
            _X = self.transformer.transform(X, y)
        else:
            _X = self.transformer.transform(X)
        return _X
