from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd
from hydrosdk.contract import ProfilingType
from hydrosdk.modelversion import ModelVersion
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder


class TransformationType(Enum):
    ONE_HOT = 0
    ORDINAL = 1
    ROBUST = 2
    NO_TRANSFORMATION = 3
    IGNORE = 4


NUMERICAL_TRANSFORMS = set(TransformationType) - {TransformationType.ONE_HOT}

PROFILE_TYPE_TO_TRANSFORMATION = {ProfilingType.NONE: TransformationType.IGNORE,
                                  ProfilingType.CATEGORICAL: TransformationType.ONE_HOT,
                                  ProfilingType.NOMINAL: TransformationType.ONE_HOT,
                                  ProfilingType.ORDINAL: TransformationType.ORDINAL,
                                  ProfilingType.NUMERICAL: TransformationType.ROBUST,
                                  ProfilingType.CONTINUOUS: TransformationType.ROBUST,
                                  ProfilingType.INTERVAL: TransformationType.NO_TRANSFORMATION,
                                  ProfilingType.RATIO: TransformationType.NO_TRANSFORMATION,
                                  ProfilingType.IMAGE: TransformationType.IGNORE,
                                  ProfilingType.VIDEO: TransformationType.IGNORE,
                                  ProfilingType.AUDIO: TransformationType.IGNORE,
                                  ProfilingType.TEXT: TransformationType.IGNORE}

NOT_IGNORED_PROFILE_TYPES = [k for k, v in PROFILE_TYPE_TO_TRANSFORMATION.items() if v != TransformationType.IGNORE]


class AutoEmbeddingsEncoder:
    """
    Class that implements interface for transforming model data into autoembeddings baased on model inputs profile.
    It implements sklearn interface fit(), fit_transform(), transform().
    """

    def __init__(self, one_hot_encoder: OneHotEncoder = None, ordinal_encoder: OrdinalEncoder = None,
                 robust_scaler: RobustScaler = None):
        """

        :param one_hot_encoder:
        :param ordinal_encoder:
        :param robust_scaler:
        """
        if one_hot_encoder is None or not one_hot_encoder.sparse or one_hot_encoder.handle_unknown != 'ignore':
            self.one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        else:
            self.one_hot_encoder = one_hot_encoder
        if ordinal_encoder is None:
            self.ordinal_encoder = OrdinalEncoder()
        else:
            self.ordinal_encoder = ordinal_encoder
        if robust_scaler is None:
            self.robust_scaler = RobustScaler()
        else:
            self.robust_scaler = robust_scaler

    def __fit__(self, transformation_type: TransformationType, features: np.array):
        """
        Fits features for only one transformation type
        :param transformation_type:
        :param features:
        :return:
        """
        if transformation_type == TransformationType.ONE_HOT:
            self.one_hot_encoder.fit(features)
        if transformation_type == TransformationType.ORDINAL:
            self.ordinal_encoder.fit(features)
        if transformation_type == TransformationType.ROBUST:
            self.robust_scaler.fit(features)


    def __transform__(self, transformation_type: TransformationType, features: np.array) -> np.array:
        """
        Transforms features of certain transformation type using specific encoder/transformer
        :param transformation_type:
        :param features:
        :return:
        """
        transformed = features
        if transformation_type == TransformationType.ONE_HOT:
            try:
                transformed = self.one_hot_encoder.transform(features)
            except NotFittedError:
                transformed = self.one_hot_encoder.fit_transform(features)
        if transformation_type == TransformationType.ORDINAL:
            try:
                transformed = self.ordinal_encoder.transform(features)
            except NotFittedError:
                transformed = self.ordinal_encoder.fit_transform(features)
        if transformation_type == TransformationType.ROBUST:
            try:
                transformed = self.robust_scaler.transform(features)
            except NotFittedError:
                transformed = self.ordinal_encoder.fit_transform(features)
        return transformed

    def __fit_transform__(self, transformation_type: TransformationType, features: np.array) -> np.array:
        """
        Does fit_trainsform on features with certain transformation type
        :param transformation_type:
        :param features:
        :return:
        """
        transformed = features
        if transformation_type == TransformationType.ONE_HOT:
            transformed = self.one_hot_encoder.fit_transform(features)
        if transformation_type == TransformationType.ORDINAL:
            transformed = self.ordinal_encoder.fit_transform(features)
        if transformation_type == TransformationType.ROBUST:
            transformed = self.robust_scaler.fit_transform(features)
        return transformed

    def fit(self, feature_map: Dict[TransformationType, np.array]):
        """
        Fits features map - mapping from transformation type and corresponding features
        :param feature_map:
        :return:
        """
        for transformation_type, features in feature_map.items():
            self.__fit__(transformation_type, features)

    def fit_transform(self, feature_map: Dict[TransformationType, np.array]) -> Dict[TransformationType, np.array]:
        """
        fits and transforms feature map - mapping from transformation type and corresponding features
        :param feature_map:
        :return:
        """
        transformation_result: Dict[TransformationType, np.array] = {}
        for transformation_type, features in feature_map.items():
            transformed_features = self.__fit_transform__(transformation_type, features)
            transformation_result[transformation_type] = transformed_features
        return transformation_result

    def transform(self, feature_map: Dict[TransformationType, np.array]) -> Dict[TransformationType, np.array]:
        """
        Transforms features map - mapping from transformation type and corresponding features
        :param feature_map:
        :return:
        """
        transformation_result: Dict[TransformationType, np.array] = {}
        for transformation_type, features in feature_map.items():
            transformed_features = self.__transform__(transformation_type, features)
            transformation_result[transformation_type] = transformed_features
        return transformation_result


def dataframe_to_feature_map(inputs_dataframe: pd.DataFrame, model: ModelVersion) -> Optional[Dict[
    TransformationType, np.array]]:
    """
    Parses dataframe data to feature map
    Feature map - mapping from transformation type and correspinding features that will be transformed using this transformation type.
    To define transformation type inputs profiles are used. Mapping ProfileType -> TransformationType is defined in PROFILE_TYPE_TO_TRANSFORMATION dict
    :param inputs_dataframe: Dataframe with model inputs as columns
    :param model: hydrosphere model version id
    :return: feature map:  Dict[TransformationType, np.array]
    """
    model_inputs = list(model.contract.predict.inputs)
    scalar_inputs = list(filter((lambda inpt: len(inpt.shape.dim) == 0), model_inputs))
    if not scalar_inputs:
        return None
    features_map: Dict[TransformationType, np.array] = {}
    for scalar_input in scalar_inputs:
        input_training_data = inputs_dataframe.get(scalar_input.name)
        if input_training_data is None:
            continue
        profiling_type = ProfilingType(scalar_input.profile)
        input_transformation = PROFILE_TYPE_TO_TRANSFORMATION.get(profiling_type, TransformationType.IGNORE)
        transformation_features = features_map.get(input_transformation, None)
        if transformation_features is None:
            transformation_features = input_training_data.to_numpy().reshape((-1, 1))
        else:
            transformation_features = np.concatenate(
                [transformation_features, input_training_data.to_numpy().reshape((-1, 1))], axis=1)
        features_map[input_transformation] = transformation_features
    return features_map
