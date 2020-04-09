## Test Demo
1. set environment variables: AWS_ACCESS_KEY, AWS_SECRET_KEY
2. from folder test: 
``sh test.sh``
2. send requests

## Requests demo: 

1. POST /visualization/plottable_embeddings/umap

```json
{        "model_name": "adult_scalar_test",
         "model_version": 3,
         "visualization_metrics": ["global_score", "sammon_error", "auc_score", "stability_score", "msid", "clustering"]
}
```
2.**POST** /visualization/jobs/<method>

    request_json:
```json
{        "model_name": "adult_scalar_test",
         "model_version": 3,
         "visualization_metrics": ["global_score", "sammon_error", "auc_score", "stability_score", "msid", "clustering"]
}
```
3. **GET** /visualization/jobs?task_id=22e86484-7d90-49fd-a3e1-329b978ee18c
