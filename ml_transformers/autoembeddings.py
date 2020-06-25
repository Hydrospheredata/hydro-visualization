from enum import Enum
from typing import Dict

import numpy as np
import umap
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder


class TransformationType(Enum):
    ONE_HOT = 0
    ORDINAL = 1
    ROBUST = 2
    NO_TRANSFORMATION = 3
    IGNORE = 4


PROFILE_TYPE_TO_TRANSFORMATION = {0: TransformationType.IGNORE,
                                  1: TransformationType.ONE_HOT,
                                  11: TransformationType.ONE_HOT,
                                  12: TransformationType.ORDINAL,
                                  2: TransformationType.ROBUST,
                                  21: TransformationType.ROBUST,
                                  22: TransformationType.NO_TRANSFORMATION,
                                  23: TransformationType.NO_TRANSFORMATION,
                                  3: TransformationType.IGNORE,
                                  4: TransformationType.IGNORE,
                                  5: TransformationType.IGNORE,
                                  6: TransformationType.IGNORE}


class AutoEmbeddingsEncoder:

    def __init__(self, one_hot_encoder: OneHotEncoder = None, ordinal_encoder: OrdinalEncoder = None,
                 robust_scaler: RobustScaler = None):
        self.updated = False
        self.fitted_one_hot, self.fitted_ordinal, self.fitted_robust = False, False, False
        if one_hot_encoder is None or not one_hot_encoder.sparse or one_hot_encoder.handle_unknown != 'ignore':
            self.one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self.updated = True
        else:
            self.one_hot_encoder = one_hot_encoder
            self.fitted_one_hot = True
        if ordinal_encoder is None:
            self.ordinal_encoder = OrdinalEncoder()
            self.updated = True
        else:
            self.ordinal_encoder = ordinal_encoder
            self.fitted_ordinal = True
        if robust_scaler is None:
            self.robust_scaler = RobustScaler()
            self.updated = True
        else:
            self.robust_scaler = robust_scaler
            self.fitted_robust = True
        self.fitted = self.fitted_one_hot and self.fitted_ordinal and self.fitted_robust

    def _fit(self, transformation_type: TransformationType, features: np.array):
        if transformation_type == TransformationType.ONE_HOT:
            self.one_hot_encoder.fit(features)
            self.fitted_one_hot = True
            self.updated = True
        if transformation_type == TransformationType.ORDINAL:
            self.ordinal_encoder.fit(features)
            self.fitted_ordinal = True
            self.updated = True
        if transformation_type == TransformationType.ROBUST:
            self.robust_scaler.fit(features)
            self.fitted_robust = True
            self.updated = True

    def _transform(self, transformation_type: TransformationType, features: np.array) -> np.array:
        transformed = features
        if transformation_type == TransformationType.ONE_HOT:
            try:
                transformed = self.one_hot_encoder.transform(features)
            except NotFittedError:
                self.need_safe = True
                transformed = self.one_hot_encoder.fit_transform(features)
        if transformation_type == TransformationType.ORDINAL:
            try:
                transformed = self.ordinal_encoder.transform(features)
            except NotFittedError:
                self.need_safe = True
                transformed = self.ordinal_encoder.fit_transform(features)
        if transformation_type == TransformationType.ROBUST:
            try:
                transformed = self.robust_scaler.transform(features)
            except NotFittedError:
                self.need_safe = True
                transformed = self.ordinal_encoder.fit_transform(features)
        return transformed

    def _fit_transform(self, transformation_type: TransformationType, features: np.array) -> np.array:
        transformed = features
        if transformation_type == TransformationType.ONE_HOT:
            transformed = self.one_hot_encoder.fit_transform(features)
            self.fitted_one_hot = True
            self.updated = True
        if transformation_type == TransformationType.ORDINAL:
            transformed = self.ordinal_encoder.fit_transform(features)
            self.fitted_ordinal = True
            self.updated = True
        if transformation_type == TransformationType.ROBUST:
            transformed = self.robust_scaler.fit_transform(features)
            self.fitted_robust = True
            self.updated = True
        return transformed

    def fit(self, feature_map: Dict[TransformationType, np.array]):
        for transformation_type, features in feature_map.items():
            self._fit(transformation_type, features)

    def fit_transform(self, feature_map: Dict[TransformationType, np.array]) -> Dict[TransformationType, np.array]:
        transformation_result: Dict[TransformationType, np.array] = {}
        for transformation_type, features in feature_map.items():
            transformed_features = self._fit_transform(transformation_type, features)
            transformation_result[transformation_type] = transformed_features
        return transformation_result

    def transform(self, feature_map: Dict[TransformationType, np.array]) -> Dict[TransformationType, np.array]:
        transformation_result: Dict[TransformationType, np.array] = {}
        for transformation_type, features in feature_map.items():
            transformed_features = self._transform(transformation_type, features)
            transformation_result[transformation_type] = transformed_features


embedding = umap.umap_.simplicial_set_embedding(fit1._raw_data, new_graph, fit1.n_components,
                                                fit1.initial_alpha, fit1._a, fit1._b,
                                                fit1.repulsion_strength, fit1.negative_sample_rate,
                                                200, fit1.init, np.random, fit1.metric,
                                                fit1._metric_kwds, False)