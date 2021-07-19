from .utils.conf import  MONGO_URL, MONGO_PORT, MONGO_AUTH_DB, MONGO_PASS, MONGO_USER
from celery import Celery


connection_string = f"mongodb://{MONGO_URL}:{MONGO_PORT}"
if MONGO_USER is not None and MONGO_PASS is not None:
    connection_string = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_URL}:{MONGO_PORT}"


celery_app = Celery(
    "celery",
    backend=f"{connection_string}/celery_broker?authSource={MONGO_AUTH_DB}",
    broker=f"{connection_string}/celery_backend?authSource={MONGO_AUTH_DB}"
)

celery_app.autodiscover_tasks(["hydro_viz.transformation_tasks.tasks"], force=True)