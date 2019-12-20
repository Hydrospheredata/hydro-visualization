# hydro-vis
Service for visualisation of high dimensional for hydrosphere

## API

1.**GET** /plottable_embeddings/transformer

    transformer - manifold learning transformer from ["umap", "trimap", "tsne"]. For now only ["umap"].
  
   request json:   
```json
{"model_name": "PACS",
 "model_version": "1",
 "data": { "bucket": "hydro-vis",
           "requests_files": ["PACS/data/requests.csv"],
           "profile_file": ""
           },
"visualization_metrics": ["global_score", "sammon_error", "auc_score", "stability_score", "msid", "clustering"]
 }
```
    
   response json:
```json
{"data_shape": [1670, 2],
 "data": "[3.1603233814,8.8767299652,2.7681264877, …]",
 "class_labels": {
                 "confidences": [0.1, 0.2, 0.3],
                 "predicted": [1, 2, 1, 2],
                 "ground_truth": [1, 1, 1, 2]
                   },
 "metrics": {
             "anomality": {
                           "scores": [0.1, 0.2, 0.5, 0.2],
                           "threshold": 0.5
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
    
2. **POST** /set_params
  
    **request format**:
    ```json
   {
   "model_name": "efficientnet",
   "model_version": "12",
   "transofmer": "umap",
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
    

### Visualize steps
1. Get embeddings from model/db, get saved transformer
    1. if there is none, create embeddings using model and default transformer
2. Compare parameters with existing saved transformer
    3. if differ fit new transformer
    4. if no transformer existed before: create new
3. If new requests are available, add them and refit transformer.
4. Send embeddings, labels, confidence, top_100 neighbours


## Demo
1. set environment variables: AWS_ACCESS_KEY, AWS_SECRET_KEY
2. send request 

GET /plottable_embeddings/transformer 

```json
{        "model_name": "PACS",
         "model_version": "1",
         "data": { "bucket": "hydro-vis",
                   "requests_files": ["PACS/data/requests.csv"],
                   "profile_file": ""
                   },
         "visualization_metrics": ["global_score", "sammon_error", "auc_score", "stability_score", "msid", "clustering"]
}
 
```

### Time usage

- receive embeddings 5s
- transform embeddigns 2-4s
- find 100 neighbours 5s

Total request handling: 12-17s