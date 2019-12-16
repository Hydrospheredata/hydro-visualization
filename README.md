# hydro-vis
Service for visualisation of high dimensional for hydrosphere


## API

1. **GET**  `/visuzalize/{transformer}`, where 
    transformer - manifold learning transformer from ["umap", "trimap", "tsne"]. For now only "umap" is available.
    
    input parameters:
    
        - model_name: str
        - model_version: int
        
    returns:
        - data_shape: tuple(n_rows, n_dimensions)
        - data: list
        - class_labels
            - ground_truth: list
            - predicted: list
            - confidences: list[float]
        - anomaly_labels
            - anomaly_labels: list[int]
            - anomaly_confidence: list[float]
        - top_100 neighbours indexes: list[list[int]]
        
2. **POST** `/set_params`
    
    parameters:
    
        - model_name: str
        - model_version: int
        - parameters: dict
    
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
