import os
from datetime import datetime

from flask import Flask, request, jsonify
from hydro_serving_grpc.reqstore import reqstore_client
from loguru import logger
from pymongo import MongoClient

from client import HydroServingClient
from data_management import S3Manager, save_record
from data_management import get_record, get_training_embeddings, get_requests_data
from ml_transformers.utils import DEFAULT_PARAMETERS
from visualizer import visualize_high_dimensional

app = Flask(__name__)
REQSTORE_URL = os.getenv("REQSTORE_URL", "managerui:9090")
SERVING_URL = os.getenv("SERVING_URL", "managerui:9090")

MONGO_URL = os.getenv("MONGO_URL", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")

hs_client = HydroServingClient(SERVING_URL)
rs_client = reqstore_client.ReqstoreClient(REQSTORE_URL, insecure=True)


def get_mongo_client():
    return MongoClient(host=MONGO_URL, port=MONGO_PORT, maxPoolSize=200,
                       username=MONGO_USER, password=MONGO_PASS,
                       authSource=MONGO_AUTH_DB)


mongo_client = get_mongo_client()

db = mongo_client['visualization']
s3manager = S3Manager()


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
               "requests_file": "PACS/data/requests.parquet",
               "profile_file": "PACS/data/requests.parquet",
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
    "requests_ids": []
    }
    '''
    start = datetime.now()
    request_json = request.get_json()
    logger.info(f'Received request: {request_json}')
    model_name = request_json.get('model_name', '')
    model_version = request_json.get('model_version', '')
    bucket_name = request_json.get('data', {}).get('bucket', '')
    requests_file = request_json.get('data', {}).get('requests_file', '')
    profile_file = request_json.get('data', {}).get('profile_file', '')
    vis_metrics = request_json.get('visualization_metrics', {})

    try:
        model = hs_client.get_model(model_name, model_version)
        servable = hs_client.deploy_servable(model_name, model_version)
    except ValueError as e:
        return jsonify({"message": f"Unable to found {model_name}v{model_version}. Error: {e}"}), 404
    except Exception as e:
        return jsonify({"message": f"Error creating model {model_name}v{model_version}. Error: {e}"}), 500

    db_record = get_record(db, method, model_name, str(model_version))
    parameters = db_record.get('parameters', {})
    results_bucket = db_record.get('embeddings_bucket_name', '')
    transfomer_path = db_record.get('transformer_file', '')
    result_path = db_record.get('result_file', '')
    if result_path:
        result = s3manager.read_json(bucket_name=results_bucket, filename=result_path)
        return result
    if transfomer_path and results_bucket:
        transformer_instance = s3manager.read_transformer(bucket_name=results_bucket, filename=transfomer_path)
    else:
        transformer_instance = None

    training_df = s3manager.read_parquet(bucket_name=bucket_name, filename=profile_file)
    requests_df = s3manager.read_parquet(bucket_name=bucket_name, filename=requests_file)
    requests_data, requests_embeddings = get_requests_data(requests_df, model.monitoring_models())
    if requests_embeddings is None:
        return jsonify({"message": f"Unable to get requests embeddings from {requests_file}"}), 404
    logger.info(f'Parsed requests data {requests_embeddings.shape}')
    training_embeddings = get_training_embeddings(model, servable, training_df)
    logger.info(f'Parsed training data, result: {training_embeddings.shape}')

    result, ml_transformer = visualize_high_dimensional(method, parameters,
                                                        training_embeddings, requests_embeddings,
                                                        transformer_instance,
                                                        vis_metrics=vis_metrics)

    result.update(requests_data)
    logger.info(f'Request handled in {datetime.now() - start}')

    # saving to S3
    if 'embedding' not in training_df.columns:
        training_df['embedding'] = training_embeddings.tolist()
        s3manager.write_parquet(training_df, bucket_name, profile_file)

    result_path = os.path.join(os.path.dirname(profile_file), f'transformed_{method}.json')
    res = s3manager.write_json(data=result, bucket_name=bucket_name,
                               filename=result_path)
    if res:
        db_record["result_file"] = result_path

    transformer_path = os.path.join(os.path.dirname(profile_file), f'{method}_transformer')
    res = s3manager.write_transformer(ml_transformer, bucket_name, transformer_path)
    if res:
        db_record['transformer_file'] = transformer_path

    save_record(db, method, db_record)
    servable.delete()
    return jsonify(result)


@app.route('/set_params/<method>', methods=['POST'])
def set_params(method):
    '''
    1. write new params to db
    :return: 200
    '''
    logger.info("Received set params request")
    request_json = request.get_json()
    model_name = request_json['model_name']
    model_version = request_json['model_version']
    parameters = request_json['parameters']
    use_labels = request_json.get('use_label', False)
    status = set_record(db, model_name, model_version, method, parameters, use_labels)
    return jsonify({}), 200, {"status": status}


def set_record(db, model_name, model_version, method, parameters, use_labels):
    '''
    Create default record with parameters for transformer
    if record exists, sets new parameters
    :param db:
    :param model_name:
    :param model_version:
    :param method:
    :param parameters:
    :param use_labels:
    :return: status {created, existed, modified}
    '''
    status = "created"
    new_method_record = {"model_name": model_name,
                         "model_version": model_version,
                         "embeddings_bucket_name": "",
                         "result_file": "",
                         "transformer_file": "",
                         "parameters": parameters,
                         "use_labels": use_labels}

    existing_record = db[method].find_one({"model_name": model_name,
                                           "model_version": model_version})

    if not parameters:
        parameters = DEFAULT_PARAMETERS[method]

    if existing_record is not None:
        existing_parameters = existing_record.get("parameters", {})
        if parameters == existing_parameters:
            logger.info("Transformer with same parameters already existed")
            status = "existed"
            return status
        else:
            db[method].update_one({"model_name": model_name,
                                   "model_version": model_version}, {"$set": new_method_record})
            logger.info("Transformer was modified")
            status = "modified"
            return status

    db[method].insert_one(new_method_record)
    return status,


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
