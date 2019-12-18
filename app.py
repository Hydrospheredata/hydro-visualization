from flask import Flask, request, jsonify
from data_management import S3Manager
from loguru import logger
from datetime import datetime
from visualizer import visualize_high_dimensional
app = Flask(__name__)

@app.route('/visualize/<method>', methods=['GET'])
def visualize(method):
    '''
    :param method: umap/trimap/tsne
    request json:
        {"model_name": "PACS",
         "model_version": "1",
         "data": { "bucket": "hydro-vis",
                   "requests_files": ["PACS/data/requests.csv"],
                   "profile_file": "data.parquet"
                   }
         }
    :return: json with:
                - data_shape (N, n_dimensions)
                - data
                - class_labels
                    - ground_truth
                    - predicted
                    - confidences
                - metric_scores
                    - anomaly_score [optional]
                - top_100
    '''
    start = datetime.now()
    request_json = request.get_json()
    # print(request_json)
    logger.info(f'Received request: {request_json}')
    model_name = request_json['model_name']
    model_version = request_json['model_version']
    bucket_name = request_json.get('data', {})['bucket']
    requests_files = request_json.get('data', {})['requests_files']
    profile_file = request_json.get('data', {})['requests_files']
    result = visualize_high_dimensional(model_name, model_version, method, bucket_name, requests_files, profile_file)
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