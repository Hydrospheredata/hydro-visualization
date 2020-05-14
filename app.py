import json
import sys

import git
from celery import Celery
from flask import Flask, request, jsonify
from flask_cors import CORS
from jsonschema import Draft7Validator
from loguru import logger as logging

from conf import MONGO_URL, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_AUTH_DB, DEBUG_ENV, \
    APP_PORT, TaskStates
from data_management import S3Manager, update_record, \
    get_mongo_client
from data_management import get_record
from ml_transformers.utils import AVAILABLE_TRANSFORMERS

with open("version") as f:
    VERSION = f.read().strip()
    repo = git.Repo(".")
    BUILDINFO = {
        "version": VERSION,
        "gitHeadCommit": repo.active_branch.commit.hexsha,
        "gitCurrentBranch": repo.active_branch.name,
        "pythonVersion": sys.version
    }

with open('./hydro-vis-request-json-schema.json') as f:
    REQUEST_JSON_SCHEMA = json.load(f)
    validator = Draft7Validator(REQUEST_JSON_SCHEMA)

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
def hello():
    return "Hi! I am Visualization service"


@app.route(PREFIX + "/buildinfo", methods=['GET'])
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
    :return: task_id if not found
    """
    if method not in AVAILABLE_TRANSFORMERS:
        return jsonify(
            {"message": f"Transformer method {method} is  not implemented.", 'status': TaskStates.NOT_SUPPORTED}), 400

    request_json = request.get_json()
    if not validator.is_valid(request_json):
        error_message = "\n".join([error.message for error in validator.iter_errors(request_json)])
        return jsonify({"message": error_message}), 400

    logging.info(f'Received request: {request_json}')

    result = transformation_tasks.tasks.transform_task.apply_async(args=(method, request_json), queue="visualization")

    return jsonify({
        'task_id': result.task_id}), 202


@app.route(PREFIX + '/jobs/<method>', methods=['POST'])
def refit_model(method):
    """
    Starts refitting transformer model
    :params model_id: model id int
    :return: job_id
    """
    refit_transformer = request.args.get('refit_transformer', True)

    if method not in AVAILABLE_TRANSFORMERS:
        return jsonify(
            {"message": f"Transformer method {method} is  not implemented.", 'status': TaskStates.NOT_SUPPORTED}), 400

    request_json = request.get_json()
    if not validator.is_valid(request_json):
        error_message = "\n".join([error.message for error in validator.iter_errors(request_json)])
        return jsonify({"message": error_message, 'status': TaskStates.NOT_SUPPORTED}), 400
    model_name = request_json.get('model_name')
    model_version = str(request_json.get('model_version'))
    db_model_info = get_record(db, method, model_name, model_version)
    db_model_info['result_file'] = ''  # forget about old results
    if refit_transformer:
        db_model_info['transformer_file'] = ''
    update_record(db, method, db_model_info, model_name, model_version)
    result = transformation_tasks.tasks.transform_task.apply_async(args=(method, request_json), queue="visualization")
    return jsonify({
        'task_id': result.task_id}), 202


@app.route(PREFIX + '/params/<method>', methods=['POST'])
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
    record['result_file'] = ''
    record['transformer_file'] = ''
    update_record(db, method, record, model_name, model_version)
    return jsonify({}), 200


@app.route(PREFIX + '/jobs', methods=['GET'])
def model_status():
    """
    Sends model status
    :param task_id:
    :return:
    """
    task_id = request.args.get('task_id')
    task = transformation_tasks.tasks.transform_task.AsyncResult(task_id)
    response = {
        'state': task.state,
        'task_id': task_id
    }

    if task.state == 'PENDING':
        # job did not start yet, do nothing
        code = 201
        pass
    elif task.state == 'PROGRESS':
        # task is accepted by worker
        code = 202
    elif task.state == 'SUCCESS':
        # job completed, return result
        result, code = task.get()
        response.update(result)
    else:
        # something went wrong in the background job, return the exception raised
        info = task.info
        response['message'] = info['message']
        code = info['code']

    return jsonify(response), code


if __name__ == "__main__":
    app.run(debug=DEBUG_ENV, host='0.0.0.0', port=APP_PORT)
