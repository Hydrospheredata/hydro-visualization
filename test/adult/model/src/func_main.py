import hydro_serving_grpc as hs
import numpy as np
from joblib import load

clf = load('/model/files/classification_model.joblib')
scaler = load('/model/files/sacler')
one_hotter = load('/model/files/one_hotter')

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


# one_hot_features = np.stack([np.array(df['Workclass'].array), np.array(df['Education'].array),
#                             np.array(df['Marital Status'].array), np.array(df['Occupation'].array),
#                             np.array(df['Relationship'].array), np.array(df['Race'].array),
#                             np.array(df['Sex'].array), np.array(df['Capital Gain'].array),
#                             np.array(df['Capital Loss'].array), np.array(df['Country'].array)], axis=1)

def extract_value(proto):
    return np.array(proto.int64_val, dtype='int64')[0]


def predict(**kwargs):
    extracted = np.array([extract_value(kwargs[feature]) for feature in features])
    transformed = np.dstack(extracted).reshape(1, len(features))

    predicted = clf.predict(transformed)

    response = hs.TensorProto(
        int64_val=[predicted.item()],
        dtype=hs.DT_INT64,
        tensor_shape=hs.TensorShapeProto())
    transformed = extracted
    age_and_class = [[transformed[0], transformed[10]]]
    categorical = [[transformed[1], transformed[2], transformed[3],
                    transformed[4], transformed[5], transformed[6],
                    transformed[7], transformed[8], transformed[9],
                    transformed[11]]]

    scaled = scaler.transform(age_and_class)
    one_hot_f = one_hotter.transform(categorical).toarray()
    embeddings = np.concatenate([scaled, one_hot_f], axis=1)
    embs_proto = hs.TensorProto(
        double_val=embeddings.flatten(),
        dtype=hs.DT_DOUBLE,
        tensor_shape=hs.TensorShapeProto(
            dim=[hs.TensorShapeProto.Dim(size=embeddings.shape[0]), hs.TensorShapeProto.Dim(size=embeddings.shape[1])]))
    conf_proto = hs.TensorProto(
        int64_val=[1],
        dtype=hs.DT_INT64,
        tensor_shape=hs.TensorShapeProto())

    return hs.PredictResponse(outputs={"embedding": embs_proto, 'class': response, 'confidence': conf_proto})
