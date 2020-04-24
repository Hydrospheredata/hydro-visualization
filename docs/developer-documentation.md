# Hysrosphere visualization service

# DEPENDENCIES

```python
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
```


# API
Whole API description is available [here](openapi.yaml).

## Request for transformed embeddings of a model to plot

- **URL**

    /visualization/plottable_embeddings/<method>
    
- **Method**

    **POST** 
- **URL Params**
    
    **method**  - name of method to use for visualization. For now only *umap*
    
- **Data Params**

    ```json
    {        "model_name": "adult_scalar",
             "model_version": 1,
             "visualization_metrics": ["global_score", "sammon_error", "auc_score", "stability_score", "msid", "clustering"]
    }
 
    ```
   model_name, model_version - name&version of a model, that should have `embedding` field in it's outputs.
   
   visualization_metrics - metrics that are used to evaluate how good will visualization reflect your real multidimensional data in 2D/3D plot. More on visualization metrics you can find [here](#visualization-metrics) 
   
   possible visualization metrics:
   
       - global_score
       - sammon_error
       - auc_score
       - stability_score
       - msid
       - clustering
   

- **Response**

    ```json
    {"task_id":  "22e86484-7d90-49fd-a3e1-329b978ee18c"}
    ```

    Service creates task and starts working on it. It is not fast (unless you have previously transformed requests for your model and not many new requests appeared since that.)
    Usually it takes from 5 to 10 minutes, but depends on:
    
      - dimensionality of your model embeddings
      - how large is your training data
      - how fast is your model
      - what metrics have you chosen (ex. stability score is very slow as it refits data several times)
    
    To get your results, you need to send request:  **GET** /visualization/jobs?task_id={YOUR-TASK-ID}

## Jobs

Visualization jobs are Celery tasks that you can invoke or get results from. 

- **URL**

    /visualization/jobs/<method>
- **Method**

    **POST** silently invokes refitting of transformer on new produciton subsample of a given model

- **URL Params**

    **method**  - name of method to use for visualization. For now only *umap*

    **refit_tr
- **Data Params**

    ```json
    {        "model_name": "adult_scalar",
             "model_version": 1
    }
    ```
 -  **Response**
 
     ```json
        {"task_id":  "22e86484-7d90-49fd-a3e1-329b978ee18c"}
     ```
    
    Service creates task and starts working on it. To get your results, you need to send request:  **GET** /visualization/jobs?task_id={YOUR-TASK-ID}

---- 

- **URL**

    /visualization/jobs/task_id=

- **Method**
   
    **GET**  returns result of a task

- **URL Params**
    
    **task_id** - id of a task returned by POST /visualization/jobs/<method>  or POST /visualization/plottable_embeddings/<method>
   
- **Data Params**

    No
    
- **Data Params**

    No
    
- **Response**
    
    Returns state of a task and result if ready
    
    states: = ['PENDING', 'RECEIVED', 'STARTED', 'FAILURE', 'REVOKED',  'RETRY'] (Source: [Celery Docs](https://docs.celeryproject.org/en/latest/reference/celery.states.html#all-states))
    
       response_json(SUCCESS):
    ```json
    {
          "result": {"data_shape": [2, 2],
                     "data": [[0.1, 0.2], [0.3, 0.4]],
                    "request_ids": [200,2001],
                     "class_labels": {
                                     "confidence": {"data": [0.1, 0.2, 0.3],
                                                     "coloring_type":  "gradient"},
                                     "class": {"data": [1, 2, 1, 3, 1],
                                               "coloring_type":  "class",
                                               "classes":  [1, 2, 3]}
                                       },
                     "metrics": {
                                 "anomality": {
                                               "scores": [0.1, 0.2, 0.5, 0.2],
                                               "threshold": 0.5,
                                               "operation": "Eq",
                                               "coloring_type": "gradient"
                                               }
                                 },
                     "top_100": [[2, 3, 4], []],  
                     "visualization_metrics": {
                                               "global_score": 0.9,
                                               "sammon_error": 0.1,
                                               "msid_score": 200
                                               }
                     },
          "state":  "SUCCESS",
          "task_id": "22e86484-7d90-49fd-a3e1-329b978ee18c",
          "description": ""
    
    }
    ```
    
       response_json (PENDING):
    ```json
    {
        "state": "PENDING",
        "task_id": "22e86484-7d90-49fd-a3e1-329b978ee18c"
    }
    ```

 
  ## API Demo
1. set environment variables: AWS_ACCESS_KEY, AWS_SECRET_KEY
2. Upload testing [adult model](test/adult) `hs apply -f serving.yaml` and send request using simulate traffic [script](test/adult/demo/simulate_traffic.py)
3. Send request on plottable embeddings:

POST /visualization/plottable_embeddings/umap

```json
{        "model_name": "adult_scalar",
         "model_version": 1,
         "visualization_metrics": ["global_score", "sammon_error", "auc_score", "stability_score", "msid", "clustering"]
}
 
 
```

4. Get results:

 GET /visualization/jobs?task_id=22e86484-7d90-49fd-a3e1-329b978ee18c


# Transformation pipeline


# Manifold Learning Transformers

## Visualization metrics
- [Cucumber](#cucumber)