import pickle

import efficientnet.keras as efn
import hydro_serving_grpc as hs
import numpy as np
import tensorflow as tf
from efficientnet.keras import preprocess_input
from keras.models import Model

IMAGE_SIZE = 380


def extract_value(proto):
    imgs = np.array(proto.float_val, dtype='float').reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 3))
    imgs_list = []
    for img in imgs:
        imgs_list.append(preprocess_input(img))
    return np.array(imgs_list)


model = efn.EfficientNetB4(weights="/model/files/efficientnet_b4")
feature_extractor = Model(model.input, model.get_layer("avg_pool").output)
del model
global graph
graph = tf.get_default_graph()

with open('/model/files/knn_model', 'rb') as file:
    knn_model = pickle.load(file)

def predict(**kwargs):
    images = extract_value(kwargs['input'])

    with graph.as_default():
        features = feature_extractor.predict(images)

    probabilities = knn_model.predict_proba(features)
    classes = np.argmax(probabilities, axis=1)
    probas = np.max(probabilities, axis=1)
    features_proto = hs.TensorProto(
        float_val=features.flatten().tolist(),
        dtype=hs.DT_FLOAT,
        tensor_shape=hs.TensorShapeProto(
            dim=[hs.TensorShapeProto.Dim(size=features.shape[0]), hs.TensorShapeProto.Dim(size=features.shape[1])]))

    probas_proto = hs.TensorProto(
        double_val=probas.flatten().tolist(),
        dtype=hs.DT_DOUBLE,
        tensor_shape=hs.TensorShapeProto(
            dim=[hs.TensorShapeProto.Dim(size=-1), hs.TensorShapeProto.Dim(size=1)]))

    classes_proto = hs.TensorProto(
        int64_val=classes.flatten().tolist(),
        dtype=hs.DT_INT64,
        tensor_shape=hs.TensorShapeProto(
            dim=[hs.TensorShapeProto.Dim(size=-1), hs.TensorShapeProto.Dim(size=1)]))

    return hs.PredictResponse(outputs={"classes": classes_proto, "probabilities": probas_proto, "embeddings": features_proto})
