from .utils import AVAILABLE_TRANSFORMERS, DEFAULT_UMAP_PARAMETERS
from abc import ABC, abstractmethod
from .metrics import global_score, sammon_error, stability_score, auc_score, intristic_multiscale_score
from loguru import logger
from umap import UMAP


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
             _use_global=True,
             _use_sammon=True,
             _use_auc=True,
             _use_stability=True,
             _use_msid=True,
             _auc_cv=5):
        '''
        Evaluates vizualization
        :param X:
        :param _X:
        :param y:
        :param _use_global:
        :param _use_sammon:
        :param _use_auc:
        :param _use_stability:
        :param _use_msid:
        :param _auc_cv: number of splits for acc evaluation
        :return: dict of metric values
        '''
        eval_metrics = {}
        if _use_global:
            eval_metrics['global_score'] = global_score(X, _X)
        if _use_sammon:
            eval_metrics['sammon_error'] = sammon_error(X, _X)
        if _use_auc and y is not None:
            acc_result = auc_score(_X, y)
            eval_metrics['knn_acc'] = acc_result['knn_acc']
            eval_metrics['svc_acc'] = acc_result['svc_acc']
        if _use_stability:
            eval_metrics['stability'] = stability_score(X, self.__instance__)
        if _use_msid:
            eval_metrics['msid'] = intristic_multiscale_score(X, _X)
        return eval_metrics



class UmapTransformer(Transformer):
    def __init__(self, parameters):
        self.default_parameters = DEFAULT_UMAP_PARAMETERS
        super().__init__(parameters)
        self.embedding_ = None

    def __set_params__(self, parameters):
        self.min_dist = parameters.get('min_dist', DEFAULT_UMAP_PARAMETERS['min_dist'])
        self.n_neighbours = parameters.get('n_neighbours', DEFAULT_UMAP_PARAMETERS['n_neighbours'])
        self.metric = parameters.get('metric', DEFAULT_UMAP_PARAMETERS['metric'])
        self.n_components = parameters.get('n_components', DEFAULT_UMAP_PARAMETERS['n_components'])
        self.use_labels = parameters.get('use_labels', False)

    def __create__(self):
        transformer = UMAP(n_neighbors=self.n_neighbours,
                           n_components=self.n_components,
                           min_dist=self.min_dist,
                           metric=self.metric)
        __instance__ = UMAP(n_neighbors=self.n_neighbours,   # unfitted instance with same parameters for
                            n_components=self.n_components,  # stability metric
                            min_dist=self.min_dist,
                            metric=self.metric)
        return transformer, __instance__

    def fit(self, X, y=None):
        if y is None:
            self.transformer.fit(X)
        elif self.use_labels:
            self.transformer.fit(X, y)
        else:
            self.transformer.fit(X)
        self.embedding_ = self.transformer.embedding_

    def fit_transform(self, X, y=None):
        if self.use_labels:
            _X = self.transformer.fit_transform(X, y)
        else:
            _X = self.transformer.fit_transform(X)
        self.embedding_ = _X
        return _X

    def transform(self, X, y=None):
        _X = self.transformer.transform(X, y)
        return _X
