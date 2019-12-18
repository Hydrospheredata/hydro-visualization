from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from scipy.spatial import procrustes
from msid.msid import msid_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import hdbscan

def global_score(X, Y):
    """
    Global score
    Input
    ------
    X: Instance matrix
    Y: Embedding
    """

    def global_loss_(X, Y):
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)
        A = X.T @ (Y @ np.linalg.inv(Y.T @ Y))
        return np.mean(np.power(X.T - A @ Y.T, 2))

    n_dims = Y.shape[1]
    Y_pca = PCA(n_components=n_dims).fit_transform(X)
    gs_pca = global_loss_(X, Y_pca)
    gs_emb = global_loss_(X, Y)
    return np.exp(-(gs_emb - gs_pca) / gs_pca)


def auc_score(X, y, cv=5, splits=None):
    '''
    return svc score, knn score
    :param splits: [(train, test)] splits from k fold splits
    '''
    result = {}
    n_classes = len(set(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [1, 10]}
    svc = svm.SVC(gamma="scale")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_tr = scaler.transform(X_train)
    X_test_tr = scaler.transform(X_test)
    if splits is not None:
        clf = GridSearchCV(svc, parameters, cv=splits, n_jobs=4)
    else:
        clf = GridSearchCV(svc, parameters, cv=cv, n_jobs=4)
    clf.fit(X_train_tr, y_train)
    y_test_predicted = clf.predict(X_test_tr)
    acc = np.mean(y_test_predicted == y_test)
    roc_auc = -1
    if n_classes == 2:
        roc_auc = roc_auc_score(y_test, y_test_predicted)
    result['svc'] = {'acc': acc, 'roc_auc': roc_auc}

    parameters = {'n_neighbors': [5, 7, 10, 15, 20], 'metric': ('euclidean', 'manhattan', 'chebyshev')}
    knn = KNN()
    clf = GridSearchCV(knn, parameters, cv=cv)
    clf.fit(X_train, y_train)
    y_test_predicted = clf.predict(X_test)
    acc = np.mean(y_test_predicted == y_test)
    roc_auc = -1
    if n_classes == 2:
        roc_auc = roc_auc_score(y_test, y_test_predicted)
    result['knn'] = {'acc': acc, 'roc_auc': roc_auc}
    return result




def stability_score(X, method, sample_indeces=None, sample_size=0.1):
    '''
    Measures stabilty of embedding by subsampling from data, fitting sample and full data and measuring Procrustes distance
    between two potentially comparable distributions
    :param  X:
    🍃
    :param transformer:
    :param sample_indeces:
    '''
    import sklearn
    if sample_indeces is not None:
        msk = sample_indeces
    else:
        msk = np.random.rand(len(X)) < sample_size
    sample = X[msk]
    embs = method.fit_transform(X)
    sample_embs = method.fit_transform(sample)
    mtx1, mtx2, disparity = procrustes(embs[msk], sample_embs)
    avg_d = np.mean(np.linalg.norm(mtx1 - mtx2, axis=1))
    return avg_d, disparity, mtx1, mtx2


def sammon_error(X, _X, distance_metric=lambda x1, x2: np.linalg.norm(x1 - x2)):
    '''
    computes sammon's error for original points in original space and points in reduced space
    X in Rn
    _X in Rm - reduced space
    X: points in original space
    _X: points in projecteed space
    distance_metric: Callable - f(x1, x2)-> float

    '''
    if len(X) != len(_X):
        raise Exception(f'Original and projection sets have different sizes: len(X)={len(X)} len(_X)={len(_X)}')
    assert len(X) == len(_X)
    orig_distances = np.array([distance_metric(X[i], X[i + 1]) for i in range(len(X) - 1)])
    proj_distances = np.array([distance_metric(_X[i], _X[i + 1]) for i in range(len(_X) - 1)])
    orig_distances += 1.e-13
    error = sum((orig_distances - proj_distances) ** 2 / orig_distances)
    error /= sum(orig_distances)
    return error


def intristic_multiscale_score(X, _X):
    '''
    Computes MSID score (https://arxiv.org/abs/1905.11141) for embeddings
    :param X:
    :param _X:
    :return:
    '''
    return msid_score(X, _X)


def clustering_scores(_X, y):
    '''
    Computes 2 scores between clustering labeling and true labels:
        - adjusted_rand_score
        - mutual_info_score
    :param X:
    :param _X:
    :param y:
    :return: adjusted_rand_score, mutual_info_score
    '''
    labels = hdbscan.HDBSCAN().fit_predict(_X) # TODO kmeans?
    mutual_info = adjusted_mutual_info_score(y, labels)
    rand_score = adjusted_rand_score(y, labels)
    return rand_score, mutual_info


