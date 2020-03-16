import os
DEBUG_ENV = bool(os.getenv("DEBUG_ENV", True))

REQSTORE_URL = os.getenv("REQSTORE_URL", "localhost:9090")  # hydro-serving.dev.hydro
SERVING_URL = os.getenv("SERVING_URL", "localhost:9090")

MONGO_URL = os.getenv("MONGO_URL", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")