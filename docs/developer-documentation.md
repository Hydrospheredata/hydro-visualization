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
Whole API description is available [here](../openapi.yaml).

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

    **refit_transformer** - if True, then transformer is refitted on new data. Otherwise, new data is inferenced using old data manifold. Default False
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
2. Upload testing [adult model](../test/adult) `hs apply -f serving.yaml` and send request using simulate traffic [script](../test/adult/demo/simulate_traffic.py)
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
Transformation consists of three main stages: 

1. Collecting model embeddings from training and production data
2. Transforming collected embeddings from N dimension to 2 dimensions
3. Caching results

## Collecting model embeddings

Model has two resources of data: training data (it is uploaded to S3 storage during model upload) and production data - all requests that go through model. 
[Transformation task](../transformation_tasks/tasks.py) starts with collecting this data. 

### Training data
First, service requests path to training data:

```
GET {CLUSTER_URL}/monitoring/training_data?modelVersionId={model.id}
```
**Method: [get_training_data_path](../transformation_tasks/tasks.py)**

However, training data usually contains only model inputs and labels, it does not have any model embeddings. To produce such embeddings we create a shadowless servable of a model and send training data as separate requests.
We do this in order to not litter unwanted requests in model monitoring. 
**Method: [compute_training_embeddings](../data_management.py)**

> If model has no training data, we ignore this step and set `training_embeddings` to `None`. This data is not required, but it is recommended to have it for more accurate transformation. 

### Production data
For visualization we do not use all produciton data, instead we request a subsample of data of size 1000. 

``
GET {CLUSTER_URL}/monitoring/checks/subsample/{model_id}?size={size}
``

Production sample is a dataframe that contains not only embeddings but also all request information. Firstly, embeddings are extracted from dataframe. Method **[parse_embeddings_from_dataframe](../data_management.py)**

> If model has no `embedding` field in it's outputs, then we cannpt extract embeddings and visualization task is stopped and return message that model has no `embedding` field. 

Secondly, we extract additional information about requests:

- class labels (if model has `class` in outputs)
- confidence scores (if model has `confidence` output)
- monitoring metrics return values, thresholds and comparison operator
- N nearest neighbours (for each request) in original embedding space
- N Closest counterfactuals (nearest requests with different labels)

**Method: [parse_requests_dataframe](../data_management.py)**

All this additional labeling data is used to color visualization. In UI you can choose how to assign colors to data points:
For continuous values (ex. confidence score) `gradient` coloring is used.

> In visualization even though monitoring metrics return continuous scores, each score is thresholded using metric threshold and comparison operator. Thus requests will be colored in only two colors.

## Transforming embeddings
Both training and production embeddings are passed to instance of manifold-learning transformer ([transformer](../ml_transformers/transformer.py)).
If transformer instance is cached, then we use method transformer.transform, which does not invoke training of transformer.
If we do not have pretrained saved transformer instance, we use transformer.fit_transform, which trains transformer with all available data.

**Method: [transform_high_dimensional](../visualizer.py)**
 
After transforming, embeddings are evaluated using specific metric that can estimate how good we fitted N-dimensional embeddings in 2-dimensional space.

## Caching results
```json
{
"model_name": "adult_scalar",
"model_version": "1",
"result_file": "s3://hydro-vis/adult_scalar/2/result.json",
"transformer_file": "s3://hydro-vis/adult_scalar/2/umap_transformer",
"parameters": {"n_neighbours": 15,
                  "min_dist": 0.1,
                  "metric":  "cosine"},
"use_labels": false
}
```

Inferencing embeddings and training transformer is time-consuming. For that we store latest results of transformation and pretrained transformer on S3 bucket `hydro-vis`. 
Path to these files are stored in mongodb. 

To visualize new requests post a job request:

**POST** /visualization/jobs/<method> 

**Data Params**

    ```json
    {        "model_name": "adult_scalar",
             "model_version": 1
    }
    ```


If a lot of new data came through model we need to refit transformer before inferencing new data. Because inferencing a lot of new data on old transformer will result in inaccurate visualization.
For this add refit_transformer parameter:


**POST** /visualization/jobs/<method>?refit_transformer=true

**Data Params**

    ```json
    {        "model_name": "adult_scalar",
             "model_version": 1
    }
    ```

### Mongodb

We use mongodb to store parameters of model visualization transformer. For each type of transformer we use separate collection. Structure of db record:

```json
{
"model_name": "adult_scalar",
"model_version": "1",
"result_file": "s3://hydro-vis/adult_scalar/2/result.json",
"transformer_file": "s3://hydro-vis/adult_scalar/2/umap_transformer",
"parameters": {"n_neighbours": 15,
                  "min_dist": 0.1,
                  "metric":  "cosine"},
"use_labels": false
}
```


# Manifold Learning Transformers


## Abstract interface
## Parametrizing transformers


## Visualization metrics


# Refitting