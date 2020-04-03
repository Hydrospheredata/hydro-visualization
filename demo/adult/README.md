# Income Classification Based on Census Data Demo
In this demo we train and deploy a classifier which takes census data (age, social class, gender, etc.) and
predicts whether person's income will be greater than 50,000$ or less.

To upload all models and instantiate applications from this demo you can run
```
dvc pull $(find . -type f -name "*.dvc")
hs apply
```

To test uploaded models you can run
```
python demo/simulate_traffic.py --secure --cluster={CLUSTER_GRPC_ADRESS}
```

## Data
We use [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/census+income) for training. For your
comfort we have uploaded it to s3 dvc cache. It can be downloaded via `dvc pull --remote s3cache data/*`

## Demo & Monitoring
We use `sklearn.ensemble.RandomForest` as a classifier model and KNN from [pyod](https://github.com/yzhao062/pyod) package as a monitoring model.
Monitoring model uses distance to nearest neighbours from training dataset
 as a way to measure incoming samples outlier score.

To load the pretrained classifier and monitoring model you can call 
`dvs pull --remote s3cache model/classification_model.joblib.dvc` and `dvs pull --remote s3cache monitoring_model/monitoring_model.joblib.dvc` correspondingly. 


To run a live demo on Hydro-Serving:
 1. Deploy both classifier and monitoring model via `hs upload --dir model/ & hs upload --dir monitoring_model/`
 2. Create applications for classifier and monitoring model 
 3. Add monitoring model as a health-check metric with "Custom Model" type to classifier in a "Monitoring" tab.
 4. Run  [showcase.ipynb](demo/showcase.ipynb) notebook.