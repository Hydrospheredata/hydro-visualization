import os

from loguru import logger as logging

DEBUG_ENV = bool(os.getenv("DEBUG_ENV", True))
APP_PORT = int(os.getenv("APP_PORT", 5000))
SERVING_URL = os.getenv("SERVING_URL", "managerui:9090")
CLUSTER_URL = os.getenv("CLUSTER_URL", "http://localhost")
SECURE = os.getenv("SECURE", False)
MONGO_URL = os.getenv("MONGO_URL", "mongodb")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")
AWS_STORAGE_ENDPOINT = os.getenv('AWS_STORAGE_ENDPOINT', '')
FEATURE_LAKE_BUCKET = os.getenv('FEATURE_LAKE_BUCKET', 'feature-lake')
HYDRO_VIS_BUCKET_NAME = os.getenv('BUCKET_NAME', 'hydro-vis')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', '')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', '')
BATCH_SIZE = 10
if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    logging.warning('No AWS keys in environment variables')