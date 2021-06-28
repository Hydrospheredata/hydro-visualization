import logging
from logging.config import fileConfig
import threading

# configure global logging policy
fileConfig("app/utils/logging_config.conf")

from app.app import run_flask, run_grpc
from app.utils.conf import DEBUG_ENV


if __name__ == "__main__":
    if DEBUG_ENV:
        logging.basicConfig(level=logging.DEBUG)
    else: 
        logging.basicConfig(level=logging.INFO)
    flask_server = threading.Thread(target=run_flask)
    grpc_server = threading.Thread(target=run_grpc)
    flask_server.start()
    grpc_server.start()
