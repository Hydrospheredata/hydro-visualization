import logging
import faulthandler
from flask import request
from hydrosdk.modelversion import ModelVersion

from .grpc_app import serve
from .ml_transformers.autoembeddings import NOT_IGNORED_PROFILE_TYPES
from .ml_transformers.utils import AVAILABLE_TRANSFORMERS, DEFAULT_PROJECTION_PARAMETERS
from .utils.conf import DEBUG_ENV, APP_PORT, EMBEDDING_FIELD, GRPC_PORT, URL_PREFIX, BUILDINFO
from .utils.conf import hs_cluster, mongo_collection, params_validator, TaskStates
from .utils import data_management
from .utils.logs import disable_logging

from .transformation_tasks.tasks import transform_task

from celery import Celery
from flask import Flask
from flask_cors import CORS

faulthandler.enable()

app = Flask(__name__)
CORS(app)


@app.route(URL_PREFIX + "/health", methods=['GET'])
@disable_logging
def hello():
    return "Ok", 200


@app.route(URL_PREFIX + "/buildinfo", methods=['GET'])
@disable_logging
def buildinfo():
    return BUILDINFO


@app.route(URL_PREFIX + '/plottable_embeddings/<method>', methods=['POST'])
def transform(method: str):
    """
    Transforms model training and requests embedding data to lower space 
    for visualization (100D to 2D) using manifold learning techniques.

    request_body: see README
    """
    request_json = request.get_json()
    if 'model_version_id' not in request_json:
        return {"message": f"Expected 'model_version_id' in body."}, 400
    if method not in AVAILABLE_TRANSFORMERS:
        return {"message": f"Transformer method {method} is  not implemented."}, 400

    result = transform_task.apply_async(
        args=(method, request_json['model_version_id']), queue="visualization")
    return {'task_id': result.task_id}, 202


@app.route(URL_PREFIX + '/supported', methods=['GET'])
def supported():
    if 'model_version_id' not in set(request.args.keys()):
        return {"message": f"Expected args: 'model_version_id'. Provided args: {set(request.args.keys())}"}, 400

    model_version_id = request.args.get('model_version_id')
    try:
        mv = ModelVersion.find_by_id(hs_cluster, int(model_version_id))
    except ValueError as e:
        logging.error(f'Couldn\'t find model version {model_version_id}. Error: {e}')
        return {"supported": False, "message": f"Unable to find {model_version_id}"}, 400
    except Exception as e:
        logging.error(f'Couldn\'t check model version {model_version_id}. Error: {e}')
        return {"supported": False, "message": f"Could not check if model {model_version_id} is valid. Error: {e}"}, 400
    try:
        if not data_management.model_has_production_data(mv.id):
            return {"supported": False, "message": "No production data."}
    except Exception as e:
        logging.error(f'Couldn\'t get production subsample. Error: {e}')
        return {"supported": False, "message": "Couldn't check production data."}
    
    training_data_path = data_management.get_training_data_path(mv)
    embeddings_exist = data_management.model_has_correct_embeddings_field(mv)
    if not training_data_path and not embeddings_exist:
        return {
            "supported": False,
            "message": f"Upload training data to use projector for models without correct "
                f"{EMBEDDING_FIELD} field. {EMBEDDING_FIELD} field should return vector of shape: "
                f"[1, some_embedding_dim]."
        }, 400
    if not embeddings_exist:
        scalar_inputs_with_profile = list(data_management.get_scalar_input_fields_with_profile(mv))
        if len(scalar_inputs_with_profile) < 2:
            return {
                "supported": False,
                "message": f"Model should have at least 2 scalar fields with one of these "
                    f"profiling types: {[profiling.name for profiling in NOT_IGNORED_PROFILE_TYPES]}."
            }, 400
    return {"supported": True}, 200


@app.route(URL_PREFIX + '/params/<method>', methods=['POST'])
def set_params(method: str):
    """
    Write transformer parameters for given model into the database so that to retrieve 
    this data during inference of plottable embeddings in future
    
    request body: see README
    """
    request_json = request.get_json()
    if 'model_version_id' not in request_json:
        return {"message": f"Expected 'model_version_id' in body."}, 400

    model_version_id = request_json.get('model_version_id')
    try:
        _ = ModelVersion.find_by_id(hs_cluster, int(model_version_id))
    except ValueError as e:
        logging.error(f'Couldn\'t find model version {model_version_id}. Error: {e}')
        return {"message": f"Unable to find {model_version_id}"}, 400
    except Exception as e:
        logging.error(f'Couldn\'t check model version {model_version_id}. Error: {e}')
        return {"message": f"Could not check if model {model_version_id} is valid. Error: {e}"}, 400
    
    logging.info(f"Received set params request for modelversion_id={model_version_id}")
    if not params_validator.is_valid(request_json):
        error_message = "\n".join([error.message for error in params_validator.iter_errors(request_json)])
        return {"message": error_message}, 400

    record = data_management.get_record(mongo_collection, method, model_version_id)
    record['parameters'] = request_json['parameters']
    record['visualization_metrics'] = request_json.get(
        'visualization_metrics', DEFAULT_PROJECTION_PARAMETERS['visualization_metrics'])
    record['training_data_sample_size'] = request_json.get(
        'training_data_sample_size', DEFAULT_PROJECTION_PARAMETERS['training_data_sample_size'])
    record['production_data_sample_size'] = request_json.get(
        'training_data_sample_size', DEFAULT_PROJECTION_PARAMETERS['production_data_sample_size'])
    record['result_file'] = ''
    record['transformer_file'] = ''

    data_management.update_record(mongo_collection, method, record, model_version_id)
    return {"message": "Set parameters."}, 200


@app.route(URL_PREFIX + '/params/<method>', methods=['GET'])
def get_params(method: str):
    if 'model_version_id' not in set(request.args.keys()):
        return {"message": f"Expected args: 'model_version_id'. Provided args: {set(request.args.keys())}"}, 400

    model_version_id = request.args.get('model_version_id')
    try:
        _ = ModelVersion.find_by_id(hs_cluster, int(model_version_id))
    except ValueError as e:
        logging.error(f'Couldn\'t find model version {model_version_id}. Error: {e}')
        return {"message": f"Unable to find {model_version_id}"}, 400
    except Exception as e:
        logging.error(f'Couldn\'t check model version {model_version_id}. Error: {e}')
        return {"message": f"Could not check if model {model_version_id} is valid. Error: {e}"}, 400
    
    record = data_management.get_record(mongo_collection, method, str(model_version_id))
    result = {
        k: v for k, v in record.items() 
        if k in ['parameters', 'visualization_metrics', 
            'production_data_sample_size', 'training_data_sample_size']
    }
    return result, 200


@app.route(URL_PREFIX + '/jobs/<method>', methods=['POST'])
def refit_model(method: str):
    """
    Starts refitting transformer model
    """
    request_json = request.get_json()
    if 'model_version_id' not in request_json:
        return {"message": f"Expected 'model_version_id' in body."}, 400

    model_version_id = request_json['model_version_id']
    refit_transformer = request_json.get('refit_transformer', True)

    if method not in AVAILABLE_TRANSFORMERS:
        return {"message": f"Transformer method {method} is  not implemented."}, 400

    db_model_info = data_management.get_record(mongo_collection, method, model_version_id)
    db_model_info['result_file'] = ''  # forget about old results
    if refit_transformer:
        db_model_info['transformer_file'] = ''
    data_management.update_record(mongo_collection, method, db_model_info, model_version_id)
    result = transform_task.apply_async(
        args=(method, model_version_id), queue="visualization")
    return {'task_id': result.task_id}, 202


@app.route(URL_PREFIX + '/jobs', methods=['GET'])
def model_status():
    """
    Returns model status.
    """
    if 'task_id' not in set(request.args.keys()):
        return {"message": f"Expected args: 'task_id'. Provided args: {set(request.args.keys())}"}, 400
    
    task_id = request.args.get('task_id')
    task = transform_task.AsyncResult(task_id)
    response = {
        'state': task.state,
        'task_id': task_id
    }

    if task.state == 'PENDING':
        # job did not start yet, do nothing
        code = 200
    elif task.state == 'STARTED':
        # task is accepted by worker
        code = 200
    elif task.state == TaskStates.SUCCESS:
        # job completed, return result
        code = 200
        result = task.get()
        response.update(result)
    else:
        # something went wrong in the background job, return the exception raised
        info = task.info
        response['message'] = info.get('message')
        code = info.get('code')
    return response, code


def run_flask():
    logging.info(f"Starting http server on port ${APP_PORT}")
    if not DEBUG_ENV:
        from gevent.pywsgi import WSGIServer
        http_server = WSGIServer(('', APP_PORT), app)
        http_server.serve_forever()
    else:
        app.run(debug=True, host='0.0.0.0', port=APP_PORT, use_reloader=False)


def run_grpc():
    logging.info(f"Starting grpc server on port ${GRPC_PORT}")
    serve()

