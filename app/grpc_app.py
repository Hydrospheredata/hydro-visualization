import logging
from concurrent import futures
from typing import Dict

import grpc
from hydro_serving_grpc.interpretability.visualization.api_pb2 import FitRequest
from hydro_serving_grpc.interpretability.visualization.api_pb2_grpc import (
    VisualizationServiceServicer, add_VisualizationServiceServicer_to_server,
)
from google.protobuf.empty_pb2 import Empty
from grpc_health.v1.health_pb2 import HealthCheckResponse
from grpc_health.v1.health_pb2_grpc import HealthServicer
from grpc_health.v1.health_pb2_grpc import add_HealthServicer_to_server

from .ml_transformers.utils import AVAILABLE_TRANSFORMERS
from .transformation_tasks.tasks import perform_transform_task, TransformResult
from .utils.conf import GRPC_PORT
from .utils.conf import mongo_collection, TaskStates
from .utils import data_management


class VisServiceServicer(VisualizationServiceServicer, HealthServicer):

    def Fit(self, request: FitRequest, context) -> Empty:
        method = 'umap'
        model_version_id = request.model_version_id
        refit_transformer = request.re
        
        if method not in AVAILABLE_TRANSFORMERS:
            logging.error(f"Transformer method {method} in request for model_version {model_version_id} is  not implemented.")
            return Empty()
        
        logging.info(f'Starting: {model_version_id}, {refit_transformer}')
        db_model_info = data_management.get_record(mongo_collection, method, model_version_id)
        db_model_info['result_file'] = ''  # forget about old results
        if refit_transformer:
            db_model_info['transformer_file'] = ''
        
        data_management.update_record(mongo_collection, method, db_model_info, model_version_id)
        result: TransformResult = perform_transform_task('umap', model_version_id)
        if result.state == TaskStates.SUCCESS:
            logging.info(f'Successfully refitted data for model_version: {model_version_id}')
            return Empty()
        else:
            meta: Dict = result.meta
            logging.info(f'Couldn\'t refit data for model_version {model_version_id}. Message: {meta.get("message", "")}')
            return Empty()

    def Check(self, request, context):
        return HealthCheckResponse(status="SERVING")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = VisServiceServicer()
    add_VisualizationServiceServicer_to_server(servicer, server)
    add_HealthServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{GRPC_PORT}')
    server.start()
    logging.info(f"Server started at [::]:{GRPC_PORT}")
    server.wait_for_termination()
