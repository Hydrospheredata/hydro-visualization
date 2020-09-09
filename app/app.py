import json
import logging
import threading
from logging.config import fileConfig
from typing import List

from celery import Celery
from flask import Flask, request, jsonify
from flask_cors import CORS
from hydrosdk.cluster import Cluster
from hydrosdk.contract import ProfilingType
from hydrosdk.modelversion import ModelVersion
from jsonschema import Draft7Validator
from waitress import serve

from grpc_app import serve
from ml_transformers.autoembeddings import NOT_IGNORED_PROFILE_TYPES
from ml_transformers.utils import AVAILABLE_TRANSFORMERS, DEFAULT_PROJECTION_PARAMETERS
from utils.conf import MONGO_URL, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_AUTH_DB, DEBUG_ENV, \
    APP_PORT, HS_CLUSTER_ADDRESS, GRPC_PROXY_ADDRESS
from utils.data_management import S3Manager, update_record, \
    get_mongo_client
from utils.data_management import get_record
from utils.data_management import model_has_embeddings, get_production_subsample, get_training_data_path
from utils.logs import disable_logging

fileConfig("utils/logging_config.ini")

with open("buildinfo.json") as f:
    BUILDINFO = json.load(f)

with open('utils/hydro-vis-params-json-schema.json') as f:
    REQUEST_JSON_SCHEMA = json.load(f)
    params_validator = Draft7Validator(REQUEST_JSON_SCHEMA)

mongo_client = get_mongo_client(MONGO_URL, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_AUTH_DB)

db = mongo_client['visualization']

s3manager = S3Manager()

app = Flask(__name__)
PREFIX = '/visualization'
CORS(app)

connection_string = f"mongodb://{MONGO_URL}:{MONGO_PORT}"
if MONGO_USER is not None and MONGO_PASS is not None:
    connection_string = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_URL}:{MONGO_PORT}"
app.config['CELERY_BROKER_URL'] = f"{connection_string}/celery_broker?authSource={MONGO_AUTH_DB}"
app.config['CELERY_RESULT_BACKEND'] = f"{connection_string}/celery_backend?authSource={MONGO_AUTH_DB}"


def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


celery = make_celery(app)
celery.autodiscover_tasks(["transformation_tasks"], force=True)
celery.conf.update({"CELERY_DISABLE_RATE_LIMITS": True})

import transformation_tasks


@app.route(PREFIX + "/health", methods=['GET'])
@disable_logging
def hello():
    return "Hi! I am Visualization service"


@app.route(PREFIX + "/buildinfo", methods=['GET'])
@disable_logging
def buildinfo():
    return jsonify(BUILDINFO)


@app.route(PREFIX + '/plottable_embeddings/<method>', methods=['POST'])
def transform(method: str):
    """
    transforms model training and requests embedding data to lower space for visualization (100D to 2D)
    using manifold learning techniques
    :param method: umap /trimap /tsne
            request body: see README
            response body: see README
    :param modelVersionId: int
    :return: task_id if not found
    """
    request_json = request.get_json()
    if 'model_version_id' not in request_json:
        return jsonify(
            {"message": f"Expected 'model_version_id' in body."}), 400

    model_version_id = request_json['model_version_id']

    if method not in AVAILABLE_TRANSFORMERS:
        return jsonify(
            {"message": f"Transformer method {method} is  not implemented."}), 400

    result = transformation_tasks.tasks.transform_task.apply_async(args=(method, model_version_id),
                                                                   queue="visualization")

    return jsonify({
        'task_id': result.task_id}), 202


@app.route(PREFIX + '/supported', methods=['GET'])
def supported():
    if 'model_version_id' not in set(request.args.keys()):
        return jsonify(
            {"message": f"Expected args: 'model_version_id'. Provided args: {set(request.args.keys())}"}), 400

    model_version_id = request.args.get('model_version_id')
    try:
        hs_cluster = Cluster(HS_CLUSTER_ADDRESS, grpc_address=GRPC_PROXY_ADDRESS)
        model = ModelVersion.find_by_id(hs_cluster, int(model_version_id))
    except ValueError as e:
        logging.error(f'Couldn\'t find model {model.name}v{model.version}. Error: {e}')
        return {"supported": False, "message": f"Unable to find {model_version_id}"}, 200
    except Exception as e:
        logging.error(f'Couldn\'t check model {model.name}v{model.version}. Error: {e}')
        return {"supported": False, "message": f"Could not check if model {model_version_id} is valid. Error: {e}"}, 200
    try:
        prod_subsample = get_production_subsample(model_version_id, size=10)
        if prod_subsample.empty:
            return {"supported": False, "message": "No production data."}
    except Exception as e:
        logging.error(f'Couldn\'t get production subsample. Error: {e}')
        return {"supported": False, "message": "Couldn't check production data."}
    training_data_path = get_training_data_path(model)
    embeddings_exist = model_has_embeddings(model)
    if not training_data_path and not embeddings_exist:
        return {"supported": False,
                "message": "Upload training data to use projector for models without 'embedding' field."}
    if not embeddings_exist:
        scalar_inputs_with_profile = list(
            filter(lambda x: (ProfilingType(x.profile) in NOT_IGNORED_PROFILE_TYPES) and (len(x.shape.dim) == 0),
                   model.contract.predict.inputs))
        if len(scalar_inputs_with_profile) < 2:
            return {"supported": False,
                    "message": f"Model should have at least 2 scalar fields with one of these profiling types: {[profiling.name for profiling in NOT_IGNORED_PROFILE_TYPES]}."}

    return {"supported": True}


@app.route(PREFIX + '/params/<method>', methods=['POST'])
def set_params(method):
    """
    Write transformer parameters for given model in database
    so that to retrieve this data during inference of plottable embeddings in future
        request body: see README
    :return: 200
    """
    request_json = request.get_json()
    if 'model_version_id' not in request_json:
        return jsonify(
            {"message": f"Expected 'model_version_id' in body."}), 400

    model_version_id = request_json['model_version_id']
    logging.info("Received set params request")

    if not params_validator.is_valid(request_json):
        error_message = "\n".join([error.message for error in params_validator.iter_errors(request_json)])
        return jsonify({"message": error_message}), 400

    record = get_record(db, method, model_version_id)
    record['parameters'] = request_json['parameters']
    record['visualization_metrics']: List[str] = request_json.get('visualization_metrics',
                                                                  DEFAULT_PROJECTION_PARAMETERS[
                                                                      'visualization_metrics'])
    record['training_data_sample_size'] = request_json.get('training_data_sample_size',
                                                           DEFAULT_PROJECTION_PARAMETERS['training_data_sample_size'])
    record['production_data_sample_size'] = request_json.get('training_data_sample_size',
                                                             DEFAULT_PROJECTION_PARAMETERS[
                                                                 'production_data_sample_size'])
    record['result_file'] = ''
    record['transformer_file'] = ''

    update_record(db, method, record, model_version_id)
    return jsonify({}), 200


@app.route(PREFIX + '/params/<method>', methods=['GET'])
def get_params(method):
    if 'model_version_id' not in set(request.args.keys()):
        return jsonify(
            {"message": f"Expected args: 'model_version_id'. Provided args: {set(request.args.keys())}"}), 400
    # TODO return 400 if such model does not exist
    model_version_id = int(request.args.get('model_version_id'))
    record = get_record(db, method, str(model_version_id))
    result = {k: v for k, v in record.items() if k in ['parameters', 'visualization_metrics',
                                                       'production_data_sample_size', 'training_data_sample_size']}

    return jsonify(result), 200


@app.route(PREFIX + '/jobs/<method>', methods=['POST'])
def refit_model(method):
    """
    Starts refitting transformer model
    :params model_id: model id int
    :return: job_id
    """
    request_json = request.get_json()
    if 'model_version_id' not in request_json:
        return jsonify(
            {"message": f"Expected 'model_version_id' in body."}), 400

    model_version_id = request_json['model_version_id']
    refit_transformer = request_json.get('refit_transformer', True)

    if method not in AVAILABLE_TRANSFORMERS:
        return jsonify(
            {"message": f"Transformer method {method} is  not implemented."}), 400

    db_model_info = get_record(db, method, model_version_id)
    db_model_info['result_file'] = ''  # forget about old results
    if refit_transformer:
        db_model_info['transformer_file'] = ''
    update_record(db, method, db_model_info, model_version_id)
    result = transformation_tasks.tasks.transform_task.apply_async(args=(method, model_version_id),
                                                                   queue="visualization")
    return jsonify({
        'task_id': result.task_id}), 202


@app.route(PREFIX + '/jobs', methods=['GET'])
def model_status():
    """
    Sends model status
    :param task_id:
    :return:
    """
    if 'task_id' not in set(request.args.keys()):
        return jsonify(
            {"message": f"Expected args: 'task_id'. Provided args: {set(request.args.keys())}"}), 400
    task_id = request.args.get('task_id')
    task = transformation_tasks.tasks.transform_task.AsyncResult(task_id)
    response = {
        'state': task.state,
        'task_id': task_id
    }

    if task.state == 'PENDING':
        # job did not start yet, do nothing
        code = 200
        pass
    elif task.state == 'STARTED':
        # task is accepted by worker
        code = 200
    elif task.state == 'SUCCESS':
        # job completed, return result
        code = 200
        result = task.get()
        response.update(result)
    else:
        # something went wrong in the background job, return the exception raised
        info = task.info
        response['message'] = info['message']
        code = info['code']

    return jsonify(response), code


def run_flask():
    logging.basicConfig(level=logging.INFO)
    if not DEBUG_ENV:
        serve(app, host='0.0.0.0', port=APP_PORT)
    else:
        app.run(debug=True, host='0.0.0.0', port=APP_PORT, use_reloader = False)


def run_grpc():
    logging.basicConfig(level=logging.INFO)
    serve()


if __name__ == "__main__":
    flask_server = threading.Thread(target=run_flask)
    grpc_server = threading.Thread(target=run_grpc)
    flask_server.start()
    grpc_server.start()
