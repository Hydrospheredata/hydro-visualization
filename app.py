from flask import Flask, request
app = Flask(__name__)

@app.route('/visualize/<method>', methods=['GET'])
def visualize(method):
    '''
    1. check if

    :param method: umap/trimap/tsne
    :return: json with:
                - data_shape (N, n_dimensions)
                - data
                - class_labels
                    - ground_truth
                    - predicted
                    - confidences
                - anomaly_labels
                    - anomaly_labels
                    - anomaly_confidence
                - top_100
    '''
    model_name = request.args.get('model_name')
    model_version = request.args.get('model_version')



@app.route('/set_params/', methods=['POST'])
def set_params():
    '''
    1. write new params to db
    :return:
    '''
    method_params = request.get_json()
    return 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)