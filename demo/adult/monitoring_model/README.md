# NY taxi load prediction

This demo contains the outlier detection model trained on Census dataset.

## Folder structure:
- [Model contract](serving.yaml) - contains deployment configuration
- [Signature function](src/func_main.py) - entry point of model servable.
- [Model demo](../demo/hydroserving_demo.ipynb) - demo on how to invoke model application

## Deployment:

To use this model as a monitoring service for another model we have to follow next rules:
1. Monitoring model inputs have to be the union of the inputs and outputs of a monitored model.
2. The output of the monitoring model is the only tensor of name `value`, dtype `double` and shape `scalar`

Here is the example of this monitoring model contract:
```yaml 
contract:
  name: "predict"
  inputs:
    age:
      shape: scalar
      type: int64
      profile: numerical
    workclass:
      shape: scalar
      type: int64
      profile: numerical
    education:
      shape: scalar
      type: int64
      profile: numerical
    marital_status:
      shape: scalar
      type: int64
      profile: numerical
    occupation:
      shape: scalar
      type: int64
      profile: numerical
    relationship:
      shape: scalar
      type: int64
      profile: numerical
    race:
      shape: scalar
      type: int64
      profile: numerical
    sex:
      shape: scalar
      type: int64
      profile: numerical
    capital_gain:
      shape: scalar
      type: int64
      profile: numerical
    capital_loss:
      shape: scalar
      type: int64
      profile: numerical
    hours_per_week:
      shape: scalar
      type: int64
      profile: numerical
    country:
      shape: scalar
      type: int64
      profile: numerical
    classes:
      shape: scalar
      type: int64
      profile: numerical
  outputs:
    value:
      shape: scalar
      type: double
      profile: numerical
``` 

To deploy a model to HS use:
```commandline
hs upload
```