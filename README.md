# hydro-vis
Service for visualisation of high dimensional for hydrosphere

## DEPENDENCIES

```python
DEBUG_ENV = bool(os.getenv("DEBUG_ENV", True))

REQSTORE_URL = os.getenv("REQSTORE_URL", "managerui:9090")
SERVING_URL = os.getenv("SERVING_URL", "managerui:9090")

MONGO_URL = os.getenv("MONGO_URL", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")
```

## Assumptions:

- Model must have in it's contract **'embedding'** output
- If model returns class prediction and confidence these fields should be named **'class'** and **'confidence'** respectively
- Only data (embeddings) from requests will be visualized. Training data is used only for accurate transformation. 

## API

1.**POST** /plottable_embeddings/<transformer>

    transformer - manifold learning transformer from ["umap", "trimap", "tsne"]. For now only ["umap"].
  
   request json:   
```json
{        "model_name": "adult_scalar",
         "model_version": 1,
         "data": { "bucket": "hydro-vis",
                   "requests_file": "adult/requests.parquet",
                   "profile_file": "adult/training.parquet"
                   },
         "visualization_metrics": ["global_score", "sammon_error", "auc_score", "stability_score", "msid", "clustering"]
}
 
 
```
    
   response json:
```json
{"data_shape": [2, 2],
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
}

```

  
2. **POST** /params/<method>
  
    **request format**:
    ```json
   {
   "model_name": "efficientnet",
   "model_version": "12",
   "parameters": {"n_neighbours": 15,
                  "min_dist": 0.1,
                  "metric":  "cosine"},
   "use_labels": "true"
    }
    ```
   
   - parameters: dict of transfomer parameters. Different set of parameters for different transformer used.
   - use_labels: true if use ground truth labels from training data. Predicted labels from production data is not
   used because this will generate false map. 
   
    **response**:
    200 - Success
    



## Demo
1. set environment variables: AWS_ACCESS_KEY, AWS_SECRET_KEY
2. upload demo/adult/model and demo/adult/monitoring_model
2. send request 

POST /plottable_embeddings/umap

```json
{        "model_name": "adult_scalar",
         "model_version": 1,
         "data": { "bucket": "hydro-vis",
                   "production_data_file": "adult/requests.parquet",
                   "profile_data_file": "adult/training.parquet"
                   },
         "visualization_metrics": ["global_score", "sammon_error", "auc_score", "stability_score", "msid", "clustering"]
}
 
 
```

### Database schema 

documents: key - model_name: model_version
collection: umap, trimap, tsne

- add created field

```json
{
"model_name": "adult_scalar",
"model_version": "1",
"embeddings_bucket_name": "hydro-vis",
"result_file": "result.json",
"transformer_file": "umap_transformer",
"parameters": {"n_neighbours": 15,
                  "min_dist": 0.1,
                  "metric":  "cosine"},
"use_labels": false
}
```
transformed_embeddings - files that store transformed embeddings with labels and other monitoring numbers

transformer structure


transformed embeddings file format:
parquet

label, confidence, transformed_embedding(vec), score1, score1_thresh, score2, score2_thresh, …
 
### Time usage

- receive embeddings 5s
- transform embeddigns 2-4s
- find 100 neighbours 5s

Total request handling: 12-17s

### Visualize steps
1. Get embeddings from model/db, get saved transformer
    1. if there is none, create embeddings using model and default transformer
2. Compare parameters with existing saved transformer
    3. if differ fit new transformer
    4. if no transformer existed before: create new
3. If new requests are available, add them and refit transformer.
4. Send embeddings, labels, confidence, top_100 neighbours



TODO add time management