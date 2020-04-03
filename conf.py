import os
DEBUG_ENV = bool(os.getenv("DEBUG_ENV", True))

SERVING_URL = os.getenv("SERVING_URL", "localhost") # "hydro-serving.dev.hydrosphere.io"
CLUSTER_URL = os.getenv("CLUSTER_URL", "http://localhost:80") # "https://hydro-serving.dev.hydrosphere.io"
SECURE = os.getenv("SECURE", False)
MONGO_URL = os.getenv("MONGO_URL", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")