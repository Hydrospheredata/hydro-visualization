from flask import Flask, request, jsonify
from data_management import S3Manager
from loguru import logger
from datetime import datetime
from visualizer import visualize_high_dimensional
import os
from hydro_serving_grpc.reqstore import reqstore_client

app = Flask(__name__)
REQSTORE_URL = os.getenv("REQSTORE_URL", "managerui:9090")
SERVING_URL = os.getenv("SERVING_URL", "managerui:9090")
# hs_client = HydroServingClient(SERVING_URL)
rs_client = reqstore_client.ReqstoreClient(REQSTORE_URL, insecure=True)


@app.route("/", methods=['GET'])
def hello():
    return "Hi! I am Visualization service"


@app.route('/plottable_embeddings/<method>', methods=['GET'])
def transform(method):
    '''
    :param method: umap/trimap/tsne
    {"model_name": "PACS",
     "model_version": "1",
     "data": { "bucket": "hydro-vis",
               "requests_files": ["PACS/data/requests.csv"],
               "profile_file": ""
               },
    "visualization_metrics": ["global_score", "sammon_error", "auc_score", "stability_score", "msid", "clustering"]
     }
    :return:
    {"data_shape": [1670, 2],
     "data": "[3.1603233814,8.8767299652,2.7681264877, …]",
     "class_labels": {
                     "confidences": [0.1, 0.2, 0.3],
                     "predicted": [1, 2, 1, 2],
                     "ground_truth": [1, 1, 1, 2]
                       },
     "metrics": {
                 "anomality": {
                               "scores": [0.1, 0.2, 0.5, 0.2],
                               "threshold": 0.5
                               }
                 },
     "top_100": [[2, 3, 4], []],
     "visualization_metrics": {
                               "global_score": 0.9,
                               "sammon_error": 0.1,
                               "msid_score": 200
                               }
    }
    '''
    start = datetime.now()
    request_json = request.get_json()
    logger.info(f'Received request: {request_json}')
    model_name = request_json['model_name']
    model_version = request_json['model_version']
    bucket_name = request_json.get('data', {})['bucket']
    requests_files = request_json.get('data', {})['requests_files']
    profile_file = request_json.get('data', {})['requests_files']
    vis_metrics = request_json.get('visualization_metrics', {})
    result = visualize_high_dimensional(model_name, model_version, method,
                                        bucket_name, requests_files, profile_file,
                                        vis_metrics=vis_metrics)
    logger.info(f'Request handled in {datetime.now() - start}')
    return jsonify(result)
    # TODO database request
    # check transofrmer, and logic


@app.route('/set_params/', methods=['POST'])
def set_params():
    '''
    1. write new params to db
    :return:
    '''
    method_params = request.get_json()
    return 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)