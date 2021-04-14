import os
import time

import click
import grpc
import hydro_serving_grpc as hs
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from hydrosdk import Cluster, Application, ModelVersion
from hydrosdk.servable import Servable

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


@click.command()
@click.option('--http-endpoint', required=True, type=str)
@click.option('--grpc-endpoint', required=True, type=str)
@click.option('--application', default="adult_scalar_vis_test")
@click.option('--ssl', is_flag=False)
def main(http_endpoint, grpc_endpoint, ssl, application):
    cluster = Cluster(http_endpoint, grpc_endpoint, ssl)
    app = Application.find(cluster, application)
    predictor = app.predictor()

    data = pd.read_csv(f"{SCRIPT_PATH}/../data/validation.csv")
    dirty_data = pd.read_csv(f"{SCRIPT_PATH}/../data/dirty_data.csv")

    # Send 500 samples of normal data
    for idx, row in tqdm(data.iloc[:500].iterrows(), total=500, desc="Sending inlier data"):
        result = predictor.predict(row.astype(np.int64))


if __name__ == "__main__":
    main()