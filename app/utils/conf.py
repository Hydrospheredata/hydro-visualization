import os

DEBUG_ENV = bool(os.getenv("DEBUG_ENV", False))
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

class TaskStates:
    NOT_SUPPORTED = 'NOT_SUPPORTED'
    ERROR = 'ERROR'
    NO_DATA = 'NO_DATA'
