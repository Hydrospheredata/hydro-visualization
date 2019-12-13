# hydro-vis
Service for visualisation of high dimensional for hydrosphere


## API

1. GET /visuzalize/transformer
    transformer - manifold learning transformer from ["umap", "trimap", "tsne"]. For now only ["umap"].
    
    parameters:
    
        - model_name
        - model_version
        
    returns:
        json with:
        
                - data_shape (N, n_dimensions)
                - data
                - class_labels
                    - ground_truth
                    - predicted
                    - confidences
                - anomaly_labels
                    - anomaly_labels
                    - anomaly_confidence
                - top_100 neighbours indexes
        
2. POST /set_params
    
    parameters:
    
        - model_name
        - model_version
        - json with parameters
    
    json format:
    ```json
   {
   "model_name": "efficientnet",
   "model_version": "12",
   "transofmer": "umap",
   "parameters": {"n_neighbours": 15,
                  "min_dist": 0.1,
                  "metric":  "cosine"},
   "visualize_split": "train"
    }
    ```
    
### Visualize steps
1. Get embeddings from model/db, get saved transformer
    1. if there is none, create embeddings using model and default transformer
2. Compare parameters with existing saved transformer
    3. if differ fit new transformer
    4. if no transformer existed before: create new
3. If new requests are available, add them and refit transformer.
4. Send embeddings, labels, confidence, top_100 neighbours