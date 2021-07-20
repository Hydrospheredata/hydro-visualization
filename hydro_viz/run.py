import logging
from logging.config import fileConfig
import threading

# configure global logging policy
fileConfig("hydro_viz/utils/logging_config.conf")

from hydro_viz.app import run_flask, run_grpc
from hydro_viz.utils.conf import DEBUG_ENV


if __name__ == "__main__":
    if DEBUG_ENV:
        logging.basicConfig(level=logging.DEBUG)
    else: 
        logging.basicConfig(level=logging.INFO)
    flask_server = threading.Thread(target=run_flask)
    grpc_server = threading.Thread(target=run_grpc)
    flask_server.start()
    grpc_server.start()
