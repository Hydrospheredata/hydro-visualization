import os
import json
import logging

from pymongo import MongoClient
from jsonschema import Draft7Validator
from hydrosdk.cluster import Cluster

from .s3manager import S3Manager


class TaskStates:
    SUCCESS = 'SUCCESS'
    NOT_SUPPORTED = 'NOT_SUPPORTED'
    ERROR = 'ERROR'
    NO_DATA = 'NO_DATA'


# initialize environment variables
try:
    DEBUG_ENV = bool(os.getenv("DEBUG", False))
    APP_PORT = int(os.getenv("APP_PORT", 5000))
    GRPC_PORT = os.getenv("GRPC_PORT", 5001)
    GRPC_PROXY_ADDRESS = os.getenv("GRPC_PROXY_ADDRESS", "localhost:9090")
    HS_CLUSTER_ADDRESS = os.getenv("HTTP_PROXY_ADDRESS", "http://localhost")
    SECURE = os.getenv("SECURE", False)
    MONGO_URL = os.getenv("MONGO_URL", "mongodb")
    MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
    MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
    MONGO_USER = os.getenv("MONGO_USER")
    MONGO_PASS = os.getenv("MONGO_PASS")
    AWS_STORAGE_ENDPOINT = os.getenv('AWS_STORAGE_ENDPOINT', '')
    AWS_REGION = os.getenv('AWS_REGION', '')
    HYDRO_VIS_BUCKET_NAME = os.getenv('AWS_BUCKET', 'visualization-artifacts')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', '')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', '')
    EMBEDDING_FIELD = 'embedding'
    MINIMUM_PROD_DATA_SIZE = 10
    N_NEIGHBOURS = 100
    URL_PREFIX = os.getenv('URL_PREFIX', '/visualization')
except ValueError as e:
    logging.error(f"Couldn't read service configuration at start up. Error: {e}")
    raise e


def get_mongo_client(host, port, user, pswd, auth_db):
    return MongoClient(
        host=host, 
        port=port, 
        maxPoolSize=200,
        username=user, 
        password=pswd,
        authSource=auth_db
    )


def get_hs_cluster(http, grpc):
    return Cluster(http, grpc)


def load_configuration():
    global mongo_factory
    global mongo_client
    global mongo_collection
    global s3manager
    # global app
    # global celery
    global hs_cluster
    global hs_cluster_factory
    
    # initialize mongo client factory
    logging.info(f"Initializing mongo client with host={MONGO_URL}, port={MONGO_PORT}, user={MONGO_USER}, pass=..., auth_db={MONGO_AUTH_DB}")
    mongo_client = get_mongo_client(MONGO_URL, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_AUTH_DB)
    mongo_collection = mongo_client["visualization"]

    # initialize s3 client
    logging.info(f"Initializing S3Manager instance with bucket={HYDRO_VIS_BUCKET_NAME}, endpoint={AWS_STORAGE_ENDPOINT}, region={AWS_REGION}")
    s3manager = S3Manager(HYDRO_VIS_BUCKET_NAME, AWS_STORAGE_ENDPOINT, AWS_REGION)

    # initialize flask application
    # logging.info("Initializing Flask")
    # app = Flask(__name__)
    # CORS(app)

    # initialize celery
    # logging.info("Initializing Celery")
    # connection_string = f"mongodb://{MONGO_URL}:{MONGO_PORT}"
    # if MONGO_USER is not None and MONGO_PASS is not None:
    #     connection_string = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_URL}:{MONGO_PORT}"
    # app.config['CELERY_BROKER_URL'] = f"{connection_string}/celery_broker?authSource={MONGO_AUTH_DB}"
    # app.config['CELERY_RESULT_BACKEND'] = f"{connection_string}/celery_backend?authSource={MONGO_AUTH_DB}"

    # celery = Celery(
    #     app.import_name,
    #     backend=app.config['CELERY_RESULT_BACKEND'],
    #     broker=app.config['CELERY_BROKER_URL']
    # )
    # celery.conf.update(app.config)

    # class ContextTask(celery.Task):
    #     def __call__(self, *args, **kwargs):
    #         with app.app_context():
    #             return self.run(*args, **kwargs)
    
    # celery.Task = ContextTask
    # celery.autodiscover_tasks(["app.transformation_tasks.tasks"], force=True)
    # celery.conf.update({"CELERY_DISABLE_RATE_LIMITS": True})

    # initialize Hydrosphere client
    logging.info(f"Initializing hydrosphere client with http_endpoint={HS_CLUSTER_ADDRESS} and grpc_endpoint={GRPC_PROXY_ADDRESS}")
    hs_cluster = get_hs_cluster(HS_CLUSTER_ADDRESS, GRPC_PROXY_ADDRESS)


# initialize global clients
mongo_client = None
mongo_collection = None
s3manager = None
# app = None
# celery = None
hs_cluster = None

load_configuration()

# read local files
with open("buildinfo.json") as f:
    BUILDINFO = json.load(f)

with open('hydro_viz/utils/hydro-vis-params-json-schema.json') as f:
    REQUEST_JSON_SCHEMA = json.load(f)
    params_validator = Draft7Validator(REQUEST_JSON_SCHEMA)
