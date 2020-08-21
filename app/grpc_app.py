import logging
from concurrent import futures
from logging.config import fileConfig
from typing import Dict

import grpc
from google.protobuf.empty_pb2 import Empty
from hydro_serving_grpc.vis import VisServiceServicer, add_VisServiceServicer_to_server, RefitRequest

from ml_transformers.utils import AVAILABLE_TRANSFORMERS
from transformation_tasks.tasks import perform_transform_task, TransformResult
from utils.conf import MONGO_URL, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_AUTH_DB
from utils.data_management import get_record, get_mongo_client, update_record

fileConfig("utils/logging_config.ini")

mongo_client = get_mongo_client(MONGO_URL, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_AUTH_DB)

db = mongo_client['visualization']


class Vis(VisServiceServicer):  # TODO Empty
    def RefitModel(self, request: RefitRequest, context):
        method = 'umap'
        model_version_id = request.model_version_id
        refit_transformer = request.refit_transformer
        if method not in AVAILABLE_TRANSFORMERS:
            logging.error(f"Transformer method {method} in request for model_version {model_version_id} is  not implemented.")
            return Empty()
        logging.info(f'Starting: {model_version_id}, {refit_transformer}')
        db_model_info = get_record(db, method, model_version_id)
        db_model_info['result_file'] = ''  # forget about old results
        if refit_transformer:
            db_model_info['transformer_file'] = ''
        update_record(db, method, db_model_info, model_version_id)
        result: TransformResult = perform_transform_task('umap', model_version_id)
        if result.state == 'SUCCESS':
            logging.info(f'Successfully refitted data for model_version: {model_version_id}')
            return Empty()
        else:
            meta: Dict = result.meta
            logging.info(f'Couldn\'t refit data for model_version {model_version_id}. Message: {meta.get("message", "")}')
            return Empty()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = Vis()
    add_VisServiceServicer_to_server(servicer, server)
    GRPC_PORT = 50051
    server.add_insecure_port(f'[::]:{GRPC_PORT}')
    server.start()
    logging.info(f"Server started at [::]:{GRPC_PORT}")
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
