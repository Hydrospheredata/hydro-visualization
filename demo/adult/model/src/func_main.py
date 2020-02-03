import hydro_serving_grpc as hs
import numpy as np
from joblib import load

clf = load('/model/files/classification_model.joblib')

features = ['age',
            'workclass',
            'education',
            'marital_status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital_gain',
            'capital_loss',
            'hours_per_week',
            'country']


def extract_value(proto):
    return np.array(proto.int64_val, dtype='int64')[0]


def predict(**kwargs):
    # extracted = np.array([extract_value(kwargs[feature]) for feature in features])
    # transformed = np.dstack(extracted).reshape(1, len(features))
    # predicted = clf.predict(transformed)

    class_proto = hs.TensorProto(
        int64_val=[int(np.random.randint(0, 2))],
        dtype=hs.DT_INT64,
        tensor_shape=hs.TensorShapeProto())

    conf_proto = hs.TensorProto(
        int64_val=[1],
        dtype=hs.DT_INT64,
        tensor_shape=hs.TensorShapeProto())

    embs = np.random.random(size=(1, 100))
    embs_proto = hs.TensorProto(
        double_val=embs.flatten(),
        dtype=hs.DT_DOUBLE,
        tensor_shape=hs.TensorShapeProto(
            dim=[hs.TensorShapeProto.Dim(size=embs.shape[0]), hs.TensorShapeProto.Dim(size=embs.shape[1])]))

    return hs.PredictResponse(outputs={"embedding": embs_proto, 'label': class_proto, 'confidence': conf_proto})
