
import hydro_serving_grpc as hs
import numpy as np
def predict(**kwargs):
    probs = np.array(kwargs['probabilities'].double_val, dtype='float').reshape((-1, 1))
    embs = np.concatenate([np.random.random(size=(1, 100)) for i in range(len(probs))])
    embs_proto = hs.TensorProto(
        double_val=embs.flatten(),
        dtype=hs.DT_DOUBLE,
        tensor_shape=hs.TensorShapeProto(dim=[hs.TensorShapeProto.Dim(size=embs.shape[0]), hs.TensorShapeProto.Dim(size=embs.shape[1])]))

    return hs.PredictResponse(outputs={"embedding": embs_proto})