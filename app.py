import json
import os
import sys
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from loguru import logger as logging
from pymongo import MongoClient

from client import HydroServingClient, HydroServingModel
from data_management import S3Manager, save_record, save_model_params
from data_management import get_record, get_training_embeddings, get_requests_data
from ml_transformers.utils import AVAILABLE_TRANSFORMERS
from visualizer import transform_high_dimensional

with open("version.json") as version_file:
    BUILDINFO = json.load(version_file)  # Load buildinfo with branchName, headCommitId and version label
    BUILDINFO['pythonVersion'] = sys.version  # Augment with python runtime version

DEBUG_ENV = bool(os.getenv("DEBUG_ENV", True))

REQSTORE_URL = os.getenv("REQSTORE_URL", "managerui:9090")
SERVING_URL = os.getenv("SERVING_URL", "managerui:9090")

MONGO_URL = os.getenv("MONGO_URL", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")

hs_client = HydroServingClient(SERVING_URL)


def get_mongo_client():
    return MongoClient(host=MONGO_URL, port=MONGO_PORT, maxPoolSize=200,
                       username=MONGO_USER, password=MONGO_PASS,
                       authSource=MONGO_AUTH_DB)


mongo_client = get_mongo_client()

db = mongo_client['visualization']
s3manager = S3Manager()

app = Flask(__name__)
CORS(app, expose_headers=['location'])


@app.route("/", methods=['GET'])
def hello():
    return "Hi! I am Visualization service"


@app.route("/buildinfo", methods=['GET'])
def buildinfo():
    return jsonify(BUILDINFO)


@app.route('/plottable_embeddings/<method>', methods=['POST'])
def transform(method):
    """
    transforms model training and requests embedding data to lower space for visualization (100D to 2D)
    using manifold learning techniques
    :param method: umap /trimap /tsne
            request body: see README
            response body: see README
    """

    if method not in AVAILABLE_TRANSFORMERS:
        return jsonify({"message": f"Transformer method {method} is  not implemented."}), 404

    start = datetime.now()
    request_json = request.get_json()
    logging.info(f'Received request: {request_json}')
    model_name = request_json.get('model_name', '')
    model_version = request_json.get('model_version', '')
    data_storage_info = request_json.get('data', {})
    bucket_name = data_storage_info.get('bucket', '')
    requests_file = data_storage_info.get('requests_file', '')
    profile_file = data_storage_info.get('profile_file', '')
    vis_metrics = request_json.get('visualization_metrics', {})

    try:
        model = hs_client.get_model(model_name, model_version)
        servable = hs_client.deploy_servable(model_name, model_version)
    except ValueError as e:
        return jsonify({"message": f"Unable to found {model_name}v{model_version}. Error: {e}"}), 404
    except Exception as e:
        return jsonify({"message": f"Error creating model {model_name}v{model_version}. Error: {e}"}), 500
    if not valid_model(model):
        servable.delete()
        return jsonify({"message": f"Invalid model {model} contract: No 'embedding' field in outputs"}), 404

    db_model_info = get_record(db, method, model_name, str(model_version))
    parameters = db_model_info.get('parameters', {})
    result_bucket = db_model_info.get('embeddings_bucket_name', '')
    transfomer_path = db_model_info.get('transformer_file', '')
    result_path = db_model_info.get('result_file', '')
    if result_path and result_bucket:
        result = s3manager.read_json(bucket_name=result_bucket, filename=result_path)
        if result:
            servable.delete()
            logging.debug(f'Request handled in {datetime.now() - start}')
            return result
    if transfomer_path and result_bucket:
        transformer_instance = s3manager.read_transformer(bucket_name=result_bucket, filename=transfomer_path)
    else:
        transformer_instance = None

    # Parsing model requests and training data
    training_df = s3manager.read_parquet(bucket_name=bucket_name, filename=profile_file)
    requests_df = s3manager.read_parquet(bucket_name=bucket_name, filename=requests_file)
    requests_data, requests_embeddings = get_requests_data(requests_df, model.monitoring_models())
    if requests_embeddings is None:
        servable.delete()
        return jsonify({"message": f"Unable to get requests embeddings from s3://{bucket_name}/{requests_file}"}), 404
    logging.info(f'Parsed requests data {requests_embeddings.shape}')
    training_embeddings = get_training_embeddings(model, servable, training_df)

    result, ml_transformer = transform_high_dimensional(method, parameters,
                                                        training_embeddings, requests_embeddings,
                                                        transformer_instance,
                                                        vis_metrics=vis_metrics)
    result.update(requests_data)
    logging.info(f'Request handled in {datetime.now() - start}')

    if training_df and 'embedding' not in training_df.columns:
        training_df['embedding'] = training_embeddings.tolist()
        s3manager.write_parquet(training_df, bucket_name, profile_file)

    if not result_bucket:
        result_bucket = bucket_name
        db_model_info['embeddings_bucket_name'] = result_bucket
    result_path = os.path.join(os.path.dirname(profile_file), f'transformed_{method}_{model}.json')
    res = s3manager.write_json(data=result, bucket_name=bucket_name,
                               filename=result_path)
    if res:
        db_model_info["result_file"] = result_path

    transformer_path = os.path.join(os.path.dirname(profile_file), f'transformer_{method}_{model}')
    res = s3manager.write_transformer(ml_transformer, bucket_name, transformer_path)
    if res:
        db_model_info['transformer_file'] = transformer_path

    save_record(db, method, db_model_info)
    servable.delete()
    return jsonify(result)


@app.route('/params/<method>', methods=['POST'])
def set_params(method):
    """
    write new params to db
    :return: 200
    """
    logging.info("Received set params request")
    request_json = request.get_json()
    model_name = request_json['model_name']
    model_version = request_json['model_version']
    parameters = request_json['parameters']
    use_labels = request_json.get('use_label', False)
    status = save_model_params(db, model_name, model_version, method, parameters, use_labels)
    return jsonify({}), 200, {"status": status}


def valid_model(model: HydroServingModel) -> [bool]:
    output_names = list(map(lambda x: x['name'], model.contract.contract_dict['outputs']))
    if 'embedding' not in output_names:
        return False


if __name__ == "__main__":
    app.run(debug=DEBUG_ENV, host='0.0.0.0', port=5000)
