import os
import time

import click
import grpc
import hydro_serving_grpc as hs
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def pack_value_into_request(val):
    # Pack values into protobuff tensors
    age_tensor = hs.TensorProto(dtype=hs.DT_INT64, int64_val=[val[0]])
    workclass_tensor = hs.TensorProto(dtype=hs.DT_INT64, int64_val=[val[1]])
    edu_tensor = hs.TensorProto(dtype=hs.DT_INT64, int64_val=[val[2]])
    marital_tensor = hs.TensorProto(dtype=hs.DT_INT64, int64_val=[val[3]])
    occupation_tensor = hs.TensorProto(dtype=hs.DT_INT64, int64_val=[val[4]])
    relationship_tensor = hs.TensorProto(dtype=hs.DT_INT64, int64_val=[val[5]])
    race_tensor = hs.TensorProto(dtype=hs.DT_INT64, int64_val=[val[6]])
    sex_tensor = hs.TensorProto(dtype=hs.DT_INT64, int64_val=[val[7]])
    gain_tensor = hs.TensorProto(dtype=hs.DT_INT64, int64_val=[val[8]])
    loss_tensor = hs.TensorProto(dtype=hs.DT_INT64, int64_val=[val[9]])
    hours_tensor = hs.TensorProto(dtype=hs.DT_INT64, int64_val=[val[10]])
    country_tensor = hs.TensorProto(dtype=hs.DT_INT64, int64_val=[val[11]])

    model_spec = hs.ModelSpec(name="census")

    # Pack tensors into a request
    request = hs.PredictRequest(model_spec=model_spec, inputs={"age": age_tensor,
                                                               "workclass": workclass_tensor,
                                                               "education": edu_tensor,
                                                               "marital_status": marital_tensor,
                                                               "occupation": occupation_tensor,
                                                               "relationship": relationship_tensor,
                                                               "race": race_tensor,
                                                               "sex": sex_tensor,
                                                               "capital_gain": gain_tensor,
                                                               "capital_loss": loss_tensor,
                                                               "hours_per_week": hours_tensor,
                                                               "country": country_tensor})
    return request


@click.command()
@click.option('--cluster', required=True, type=str)
@click.option('--secure', is_flag=True)
def main(cluster, secure):
    if secure:
        channel = grpc.secure_channel(cluster, credentials=grpc.ssl_channel_credentials())
    else:
        channel = grpc.insecure_channel(cluster)
    stub = hs.PredictionServiceStub(channel)

    data = pd.read_csv(f"{SCRIPT_PATH}/../data/validation.csv")
    dirty_data = pd.read_csv(f"{SCRIPT_PATH}/../data/dirty_data.csv")

    # Send 500 samples of normal data
    for idx, row in tqdm(data.iloc[:500].iterrows(), total=500, desc="Sending inlier data"):
        x = np.array(row)
        request = pack_value_into_request(x)
        _ = stub.Predict(request)
        time.sleep(0.35)

    # Send 1000 samples of normal data
    dirty_sample = dirty_data.iloc[2500:].sample(1000)
    for idx, row in tqdm(dirty_sample.iterrows(), total=len(dirty_sample), desc="Sending outlier data"):
        x = np.array(row)
        request = pack_value_into_request(x)
        result = stub.Predict(request)
        time.sleep(0.25)


if __name__ == '__main__':
    main()
