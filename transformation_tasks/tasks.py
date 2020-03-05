from flask import jsonify

import os

import numpy as np
from flask import jsonify
from loguru import logger as logging

from app import celery, s3manager, hs_client, get_mongo_client
from client import HydroServingModel
from data_management import get_record, parse_embeddings_from_dataframe, parse_requests_dataframe, \
    compute_training_embeddings
from visualizer import transform_high_dimensional


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


@celery.task(bind=True)
def transform_task(self, method, request_json):  # todo: specify different behaviour for refitting
    mongo_client = get_mongo_client()
    db = mongo_client['visualization']

    model_name = request_json.get('model_name')
    model_version = request_json.get('model_version')
    data_storage_info = request_json.get('data')
    bucket_name = data_storage_info.get('bucket')
    requests_file = data_storage_info.get('requests_file')
    path_to_training_data = data_storage_info.get('profile_file', '')
    vis_metrics = request_json.get('visualization_metrics', {})

    db_model_info = get_record(db, method, model_name, str(model_version))
    parameters = db_model_info.get('parameters', {})
    result_bucket = db_model_info.get('embeddings_bucket_name', '')
    path_to_transformer = db_model_info.get('transformer_file', '')
    result_path = db_model_info.get('result_file', '')

    transformer = None
    try:
        model = hs_client.get_model(model_name, model_version)
    except ValueError as e:
        return {"message": f"Unable to found {model_name}v{model_version}. Error: {e}"}, 400
    except Exception as e:
        return {"message": f"Error {model_name}v{model_version}. Error: {e}"}, 500
    if not valid_embedding_model(model):
        return {"message": f"Invalid model {model} contract: No 'embedding' field in outputs"}, 404

    # Parsing model requests and training data
    training_df = s3manager.read_parquet(bucket_name=bucket_name, filename=path_to_training_data)
    production_requests_df = s3manager.read_parquet(bucket_name=bucket_name,
                                                    filename=requests_file)  # FIXME Ask Yura for subsampling code

    if 'embedding' not in production_requests_df.columns:
        return jsonify(
            {"message": f"Unable to get requests embeddings from s3://{bucket_name}/{requests_file}"}), 404

    production_embeddings = parse_embeddings_from_dataframe(production_requests_df)
    requests_data_dict = parse_requests_dataframe(production_requests_df, model.monitoring_models())
    logging.info(f'Parsed requests data shape: {production_embeddings.shape}')

    if training_df is None:
        training_embeddings = None
    elif 'embedding' in training_df.columns:
        logging.debug('Training embeddings exist')
        training_embeddings = np.stack(training_df['embedding'].values)
    else:  # infer embeddings using model
        try:
            servable = hs_client.deploy_servable(model_name, model_version)
        except ValueError as e:
            return {"message": f"Unable to found {model_name}v{model_version}. Error: {e}"}, 404
        except Exception as e:
            return {"message": f"Error {model_name}v{model_version}. Error: {e}"}, 500

        training_embeddings = compute_training_embeddings(model, servable, training_df)
        servable.delete()

    plottable_data, transformer = transform_high_dimensional(method, parameters,
                                                             training_embeddings, production_embeddings,
                                                             transformer,
                                                             vis_metrics=vis_metrics)
    plottable_data.update(requests_data_dict)

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

    return plottable_data


