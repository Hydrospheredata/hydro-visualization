# hydro-vis
Service for visualisation of high dimensional for hydrosphere


## API

1. GET /visuzalize/transformer
    transformer - manifold learning transformer from ["umap", "trimap", "tsne"]. For now only ["umap"].
    
    parameters:
        - model_name
        - model_version
        
2. POST /set_params
    
    parameters:
        - model_name
        - model_version
        - json with parameters
    
    json format:
    ```json
   {
   "transofmer": "umap",
   "parameters": {"n_neighbours": 15,
                  "min_dist": 0.1,
                  "metric":  "cosine"}
    }
    ```
    
