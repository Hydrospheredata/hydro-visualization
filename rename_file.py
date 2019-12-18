import os
from loguru import logger
import pickle


def embeddings_exist(model_name, model_version):
    '''
    TODO change names of points_path to embeddings(of model)
    :param model_name:
    :param model_version:
    :return:
    '''
    folder_name = f'data/{model_name}_{model_version}'
    if not os.path.exists(folder_name):
        logger.error(f'Data for {model_name}_{model_version} not found')
        return False
    points_path = os.path.join(folder_name, f'{model_name}_X.npy')
    labels_path = os.path.join(folder_name, f'{model_name}_y.npy')
    if not os.path.exists(points_path):
        logger.error(f'No profile data exists for {model_name}_{model_version}')
        return False
    if not os.path.exists(labels_path):
        logger.info(f'No labels found for this model')
    transfromed_emb_path = os.path.join(folder_name, f'{model_name}_embeddings_tr.npy')
    if os.path.exists(transfromed_emb_path):
        return True

    return False


def transformer_exist(model_name, model_version):
    folder_name = f'data/{model_name}_{model_version}'
    transformer_path = os.path.join(folder_name, f'transformer.pkl')
    return os.path.exists(transformer_path)


def need_refit_transformer(model_name, model_version, parameters):
    folder_name = f'data/{model_name}_{model_version}'
    transformer_path = os.path.join(folder_name, f'transformer.pkl')
    with open(transformer_path, 'rb') as file:
        transformer = pickle.load(file)

    transformer
    return False
