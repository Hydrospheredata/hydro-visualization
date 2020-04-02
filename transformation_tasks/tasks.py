import os
from datetime import datetime

import numpy as np
from hydrosdk.model import Model
from hydrosdk.monitoring import MetricSpec
from loguru import logger as logging

from app import celery, s3manager, hs_client, hs_cluster
from conf import MONGO_URL, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_AUTH_DB
from data_management import get_record, parse_embeddings_from_dataframe, parse_requests_dataframe, \
    compute_training_embeddings, update_record, get_mongo_client
from visualizer import transform_high_dimensional


def valid_embedding_model(model: Model) -> [bool]:
    """
    Check if model returns embeddings
    :param model:
    :return:
    """
    output_names = [field.name for field in model.contract.predict.outputs]
    if 'embedding' not in output_names:
        return False
    return True


@celery.task(bind=True)
def transform_task(self, method, request_json):
    start = datetime.now()
    mongo_client = get_mongo_client(MONGO_URL,  MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_AUTH_DB)
    db = mongo_client['visualization']

    model_name = request_json.get('model_name')
    model_version = str(request_json.get('model_version'))
    data_storage_info = request_json.get('data')
    bucket_name = data_storage_info.get('bucket')
    requests_file = data_storage_info.get('production_data_file')
    path_to_training_data = data_storage_info.get('profile_data_file', '')
    vis_metrics = request_json.get('visualization_metrics', {})

    db_model_info = get_record(db, method, model_name, str(model_version))
    parameters = db_model_info.get('parameters', {})
    result_bucket = db_model_info.get('embeddings_bucket_name', '')
    path_to_transformer = db_model_info.get('transformer_file', '')
    result_path = db_model_info.get('result_file', '')
    if result_path and result_bucket:
        plottable_data = s3manager.read_json(bucket_name=result_bucket, filename=result_path)
        if plottable_data:
            return {"result": plottable_data}, 200

    try:
        model = Model.find(hs_cluster, model_name, int(model_version))
    except ValueError as e:
        return {"message": f"Unable to find {model_name}v{model_version}. Error: {e}"}, 404
    except Exception as e:
        return {"message": f"Error {model_name}v{model_version}. Error: {e}"}, 500
    if not valid_embedding_model(model):
        return {"message": f"Invalid model {model} contract: No 'embedding' field in outputs"}, 404

    if path_to_transformer and result_bucket:
        transformer = s3manager.read_transformer_model(bucket_name=result_bucket, filename=path_to_transformer)
    else:
        transformer = None


    # Parsing model requests and training data   # TODO SDK
    training_df = s3manager.read_parquet(bucket_name=bucket_name, filename=path_to_training_data)
    production_requests_df = s3manager.read_parquet(bucket_name=bucket_name,
                                                    filename=requests_file)  # FIXME Ask Yura for subsampling code

    if 'embedding' not in production_requests_df.columns:
        return f"Unable to get requests embeddings from s3://{bucket_name}/{requests_file}", 404

    production_embeddings = parse_embeddings_from_dataframe(production_requests_df)
    monitoring_models_conf = [(metric.name, metric.config.threshold_op) for metric in MetricSpec.list_for_model(hs_cluster, model.id)]
    requests_data_dict = parse_requests_dataframe(production_requests_df, monitoring_models_conf, production_embeddings)


    logging.info(f'Parsed requests data shape: {production_embeddings.shape}')

    if training_df is None:
        training_embeddings = None
    elif 'embedding' in training_df.columns:  # TODO Remove
        logging.debug('Training embeddings exist')
        training_embeddings = np.stack(training_df['embedding'].values)
    else:  # infer embeddings using model
        try:
            servable = hs_client.deploy_servable(model_name, int(model_version))  # TODO SDK
        except ValueError as e:
            return {"message": f"Unable to find {model_name}v{model_version}. Error: {e}"}, 404
        except Exception as e:
            return {"message": f"Error {model_name}v{model_version}. Error: {e}"}, 500

        training_embeddings = compute_training_embeddings(model, servable, training_df)
        servable.delete()

    plottable_data, transformer = transform_high_dimensional(method, parameters,
                                                             training_embeddings, production_embeddings,
                                                             transformer,
                                                             vis_metrics=vis_metrics)
    plottable_data.update(requests_data_dict)
    plottable_data["parameters"] = parameters

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

    update_record(db, method, db_model_info, model_name, model_version)

    logging.info(f'Request handled in {datetime.now() - start}')

    return {"result": plottable_data}, 200
