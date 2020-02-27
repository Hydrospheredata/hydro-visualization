import json
import os
import sys
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from jsonschema import Draft7Validator
from loguru import logger as logging
from pymongo import MongoClient

from client import HydroServingClient, HydroServingModel
from data_management import S3Manager, parse_embeddings_from_dataframe, parse_requests_dataframe
from data_management import get_record, compute_training_embeddings
from ml_transformers.utils import AVAILABLE_TRANSFORMERS
from visualizer import transform_high_dimensional

with open("version.json") as version_file:
    BUILDINFO = json.load(version_file)  # Load buildinfo with branchName, headCommitId and version label
    BUILDINFO['pythonVersion'] = sys.version  # Augment with python runtime version

with open('./hydro-vis-request-json-schema.json') as f:
    REQUEST_JSON_SCHEMA = json.load(f)
    validator = Draft7Validator(REQUEST_JSON_SCHEMA)

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


cl = MongoClient(host='localhost', port=27017, maxPoolSize=200,
                 username=MONGO_USER, password=MONGO_PASS,
                 authSource=MONGO_AUTH_DB)

mongo_client = get_mongo_client()

db = mongo_client['visualization']
s3manager = S3Manager()

app = Flask(__name__)
CORS(app)


@app.route("/", methods=['GET'])
def hello():
    return "Hi! I am Visualization service"


@app.route("/buildinfo", methods=['GET'])
def buildinfo():
    return jsonify(BUILDINFO)


@app.route('/plottable_embeddings/<method>', methods=['POST'])
def transform(method: str):
    """
    transforms model training and requests embedding data to lower space for visualization (100D to 2D)
    using manifold learning techniques
    :param method: umap /trimap /tsne
            request body: see README
            response body: see README
    """

    if method not in AVAILABLE_TRANSFORMERS:
        return jsonify({"message": f"Transformer method {method} is  not implemented."}), 400

    request_json = request.get_json()
    if not validator.is_valid(request_json):
        error_message = "\n".join([error.message for error in validator.iter_errors(request_json)])
        return jsonify({"message": error_message}), 400

    start = datetime.now()
    logging.info(f'Received request: {request_json}')

    model_name = request_json.get('model_name', '')
    model_version = request_json.get('model_version', '')
    data_storage_info = request_json.get('data', {})
    bucket_name = data_storage_info.get('bucket', '')
    requests_file = data_storage_info.get('requests_file', '')
    path_to_training_data = data_storage_info.get('profile_file', '')
    vis_metrics = request_json.get('visualization_metrics', {})

    try:
        model = hs_client.get_model(model_name, model_version)
    except ValueError as e:
        return jsonify({"message": f"Unable to found {model_name}v{model_version}. Error: {e}"}), 404
    except Exception as e:
        return jsonify({"message": f"Error {model_name}v{model_version}. Error: {e}"}), 500
    if not valid_model(model):
        return jsonify({"message": f"Invalid model {model} contract: No 'embedding' field in outputs"}), 404

    db_model_info = get_record(db, method, model_name, str(model_version))
    parameters = db_model_info.get('parameters', {})
    result_bucket = db_model_info.get('embeddings_bucket_name', '')
    path_to_transformer = db_model_info.get('transformer_file', '')
    result_path = db_model_info.get('result_file', '')
    if result_path and result_bucket:
        plottable_data = s3manager.read_json(bucket_name=result_bucket, filename=result_path)
        if plottable_data:
            logging.info(f'Request handled in {datetime.now() - start}')
            return plottable_data
    if path_to_transformer and result_bucket:
        transformer = s3manager.read_transformer_model(bucket_name=result_bucket, filename=path_to_transformer)
    else:
        transformer = None

    # Parsing model requests and training data
    training_df = s3manager.read_parquet(bucket_name=bucket_name, filename=path_to_training_data)
    production_requests_df = s3manager.read_parquet(bucket_name=bucket_name,
                                                    filename=requests_file)  # FIXME Ask Yura for subsampling code

    if 'embedding' not in production_requests_df.columns:
        return jsonify({"message": f"Unable to get requests embeddings from s3://{bucket_name}/{requests_file}"}), 404

    production_embeddings = parse_embeddings_from_dataframe(production_requests_df)
    requests_data_dict = parse_requests_dataframe(production_requests_df, model.monitoring_models())
    logging.info(f'Parsed requests data shape: {production_embeddings.shape}')

    if training_df is not None:
        training_embeddings = None
    elif 'embedding' in training_df.columns:
        logging.debug('Training embeddings exist')
        training_embeddings = np.stack(training_df['embedding'].values)
    else:  # infer embeddings using model
        try:
            servable = hs_client.deploy_servable(model_name, model_version)
        except ValueError as e:
            return jsonify({"message": f"Unable to found {model_name}v{model_version}. Error: {e}"}), 404
        except Exception as e:
            return jsonify({"message": f"Error {model_name}v{model_version}. Error: {e}"}), 500

        training_embeddings = compute_training_embeddings(model, servable, training_df)
        servable.delete()

    plottable_data, transformer = transform_high_dimensional(method, parameters,
                                                             training_embeddings, production_embeddings,
                                                             transformer,
                                                             vis_metrics=vis_metrics)
    plottable_data.update(requests_data_dict)
    logging.info(f'Request handled in {datetime.now() - start}')

    if training_df is not None and 'embedding' not in training_df.columns:
        training_df['embedding'] = training_embeddings.tolist()
        s3manager.write_parquet(training_df, bucket_name, path_to_training_data)

    if not result_bucket:
        result_bucket = bucket_name
        db_model_info['embeddings_bucket_name'] = result_bucket

    result_path = os.path.join(os.path.dirname(path_to_training_data),
                               f'transformed_{method}_{model_name}{model_version}.json')
    s3manager.write_json(data=plottable_data, bucket_name=bucket_name,
                         filename=result_path)
    db_model_info["result_file"] = result_path

    transformer_path = os.path.join(os.path.dirname(path_to_training_data),
                                    f'transformer_{method}_{model_name}{model_version}')
    transformer_saved_to_s3 = s3manager.write_transformer_model(transformer, bucket_name, transformer_path)

    if transformer_saved_to_s3:
        db_model_info['transformer_file'] = transformer_path
    if '_id' in db_model_info:
        db[method].update_one({"model_name": model_name,
                               "model_version": model_version, "_id": str(db_model_info['_id'])},
                              {"$set": db_model_info})
    else:
        db[method].update_one({"model_name": model_name,
                               "model_version": model_version},
                              {"$set": db_model_info}, upsert=True)

    return jsonify(plottable_data)


@app.route('/params/<method>', methods=['POST'])
def set_params(method):
    """
    Write transformer parameters for given model in database
    so that to retrieve this data during inference of plottable embeddings in future
        request body: see README
    :return: 200
    """
    logging.info("Received set params request")
    request_json = request.get_json()
    model_name = request_json['model_name']
    model_version = request_json['model_version']
    parameters = request_json['parameters']
    use_labels = request_json.get('use_label', False)
    record = get_record(db, method, model_name, model_version)
    record['parameters'] = parameters
    record['use_labels'] = use_labels

    if '_id' in record:
        db[method].update_one({"model_name": model_name,
                               "model_version": model_version, "_id": str(record['_id'])},
                              {"$set": record})
    else:
        db[method].update_one({"model_name": model_name,
                               "model_version": model_version},
                              {"$set": record}, upsert=True)

    return jsonify({}), 200


def valid_model(model: HydroServingModel) -> [bool]:
    """
    Check if model returns embeddings
    :param model:
    :return:
    """

    output_names = list(map(lambda x: x['name'], model.contract.contract_dict['outputs']))
    if 'embedding' not in output_names:
        return False
    return True


if __name__ == "__main__":
    app.run(debug=DEBUG_ENV, host='0.0.0.0', port=5000)
