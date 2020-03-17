import json
import sys

from celery import Celery
from flask import Flask, request, jsonify
from flask_cors import CORS
from jsonschema import Draft7Validator
from loguru import logger as logging

from client import HydroServingClient, HydroServingModel
from conf import SERVING_URL, MONGO_URL, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_AUTH_DB, DEBUG_ENV
from data_management import S3Manager, update_record, \
    get_mongo_client
from data_management import get_record
from ml_transformers.utils import AVAILABLE_TRANSFORMERS

with open("version.json") as version_file:
    BUILDINFO = json.load(version_file)  # Load buildinfo with branchName, headCommitId and version label
    BUILDINFO['pythonVersion'] = sys.version  # Augment with python runtime version

with open('./hydro-vis-request-json-schema.json') as f:
    REQUEST_JSON_SCHEMA = json.load(f)
    validator = Draft7Validator(REQUEST_JSON_SCHEMA)

# DEBUG_ENV = bool(os.getenv("DEBUG_ENV", True))
#
# REQSTORE_URL = os.getenv("REQSTORE_URL", "localhost:9090")  # hydro-serving.dev.hydro
# SERVING_URL = os.getenv("SERVING_URL", "localhost:9090")
#
# MONGO_URL = os.getenv("MONGO_URL", "localhost")
# MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
# MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
# MONGO_USER = os.getenv("MONGO_USER")
# MONGO_PASS = os.getenv("MONGO_PASS")

hs_client = HydroServingClient(SERVING_URL)

# def get_mongo_client():
#     return MongoClient(host=MONGO_URL, port=MONGO_PORT, maxPoolSize=200,
#                        username=MONGO_USER, password=MONGO_PASS,
#                        authSource=MONGO_AUTH_DB)
#
#
# cl = MongoClient(host='localhost', port=27017, maxPoolSize=200,
#                  username=MONGO_USER, password=MONGO_PASS,
#                  authSource=MONGO_AUTH_DB)

mongo_client = get_mongo_client(MONGO_URL, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_AUTH_DB)

db = mongo_client['visualization']
s3manager = S3Manager()

app = Flask(__name__)
CORS(app)

connection_string = f"mongodb://{MONGO_URL}:{MONGO_PORT}"
if MONGO_USER is not None and MONGO_PASS is not None:
    connection_string = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_URL}:{MONGO_PORT}"
app.config['CELERY_BROKER_URL'] = f"{connection_string}/celery_broker?authSource={MONGO_AUTH_DB}"
app.config['CELERY_RESULT_BACKEND'] = f"{connection_string}/celery_backend?authSource={MONGO_AUTH_DB}"


# if MONGO_USER is not None and MONGO_PASS is not None:
# #     connection_string = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_URL}:{MONGO_PORT}"
# app.config['CELERY_BROKER_URL'] = 'pyamqp://'
# app.config['CELERY_RESULT_BACKEND'] = 'rpc://'
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
# celery.conf.update(app.config)
# celery.conf.update({"CELERY_DISABLE_RATE_LIMITS": True})

import transformation_tasks


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
    :return: task_id if not found
    """
    if method not in AVAILABLE_TRANSFORMERS:
        return jsonify({"message": f"Transformer method {method} is  not implemented."}), 400

    request_json = request.get_json()
    if not validator.is_valid(request_json):
        error_message = "\n".join([error.message for error in validator.iter_errors(request_json)])
        return jsonify({"message": error_message}), 400

    logging.info(f'Received request: {request_json}')

    result = transformation_tasks.tasks.transform_task.delay(method, request_json)

    return jsonify({
        'Task_id': result.task_id}), 202


@app.route('/jobs/<method>', methods=['POST'])
def refit_model(method):
    """
    Starts refitting transformer model
    TODO change to model_id request
    :params model_id: model id int
    :return: job_id
    """
    refit_transformer = request.args.get('refit_transformer', True)

    if method not in AVAILABLE_TRANSFORMERS:
        return jsonify({"message": f"Transformer method {method} is  not implemented."}), 400

    request_json = request.get_json()
    if not validator.is_valid(request_json):
        error_message = "\n".join([error.message for error in validator.iter_errors(request_json)])
        return jsonify({"message": error_message}), 400
    model_name = request_json.get('model_name')
    model_version = str(request_json.get('model_version'))
    db_model_info = get_record(db, method, model_name, model_version)
    db_model_info['result_file'] = ''  # forget about old results
    if refit_transformer:
        db_model_info['transformer_file'] = ''
    update_record(db, method, db_model_info, model_name, model_version)
    result = transformation_tasks.tasks.transform_task.delay(method, request_json)
    return jsonify({
        'task_id': result.task_id}), 202


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
    record['result_file'] = ''
    record['transformer_file'] = ''
    update_record(db, method, record, model_name, model_version)
    return jsonify({}), 200


@app.route('/jobs', methods=['GET'])
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
        pass
    elif task.state == 'SUCCESS':
        # job completed, return result
        result = task.get()

        response['result'] = result
    else:
        # something went wrong in the background job, return the exception raised
        response['description'] = task.info

    return jsonify(response)


def valid_embedding_model(model: HydroServingModel) -> [bool]:
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
