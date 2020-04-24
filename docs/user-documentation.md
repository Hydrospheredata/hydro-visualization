# Hysrosphere visualization service

Hydro-visualization is a service that visualizes production data in embedding space of model. This great tool allows to find more insights about data, detect domain drift and model limitations. 

# Why to use and what it does

Visualization of embedding space of your model can bring you various insights about your data and model performance. 

Embeddings are low-dimensional, learned continuous vector representations of discrete variables

embeddings can be used to:

- find nearest points (points that your model considered close to each other)
- detect domain drift
- detect data where model makes mistakes
- detect closes counterfactual - points that are close to each other but are classified by model as different

Lets see what information our service provides:

- Visualization of all production requests embeddings with various colorings:
    - Colouring based on model prediction
    - Colouring based on model confidence in predictions
    - Colouring based on scores of your monitoring models
- Closest requests to specific request
- Closest counterfactuals to specific request
- All information about request

# How to use it

## 1. Create Model and Application

Create your model, which will receive some inputs and return outputs which contain field `embedding`. Embedding should be a 1 D vector.  Upload your model using command `hs upload`. 

For more accurate transformation add training data in your serving.yaml file. You can also add monitoring model. If you do so, in visualization you can find gradient coloring of each request according to monitoring score.

## 2. Send data to your model

Send  requests to your model using  GRPC predict requests  or SDK. 

## 3. Go to UI

Got to your embeddings model In UI. In `visualization` tab you can see  visualization of your requests embeddings.

# Limitations

- only tabular data for now
- at least 1000 production requests