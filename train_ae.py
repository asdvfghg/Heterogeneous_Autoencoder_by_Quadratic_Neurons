from __future__ import division
from __future__ import print_function

import os
import random
import sys
# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
from itertools import product

import torch
from Model.HAutoEncoder import AutoEncoder

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.io import loadmat
from pyod.utils.utility import standardizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from Model.HAutoEncoder import AutoEncoder
from utils.train_function import split_data


def random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random_state = np.random.RandomState(seed)
    return random_state


if __name__ == "__main__":

    mat_file_list = [
        # 'arrhythmia.mat',
        'glass.mat',
        # 'musk.mat',
        # 'optdigits.mat',
        # 'pendigits.mat',
        # 'pima.mat',
        # 'vertebral.mat',
        # 'wbc.mat',
    ]
    seed = 42
    ae_model = 'HAE_X'  # Available parameter:  AE, QAE, HAE_X, HAE_Y, HAE_I
    # Default hyperparameters, all need to careful design for different data sets.

    learning_rate = 0.001
    batch_size = 128
    alpha = 0.05 # In AE models, sub_learning_rate is not used, but we recommend to set 0.0 for a mark.
    epochs = 100

    mat_file = mat_file_list[0]
    df_columns = ['AUC', 'Pre', 'Recall', 'F1']
    ae_result_df = pd.DataFrame(columns=df_columns)
    print("\n... Processing", mat_file, '...')
    mat = loadmat(os.path.join('./data', mat_file))
    X = mat['X']
    y = mat['y'].ravel()
    # 0-normal 1-abnormal
    auc_list = []
    pre_list = []
    recall_list = []
    f1_list = []


    random_state = random_seed(seed)
    X_train, X_test, y_train, y_test = split_data(X, y, random_state)
    outliers_fraction = np.count_nonzero(y) / len(y)
    outliers_percentage = round(outliers_fraction * 100, ndigits=4)  # 返回浮点数四舍五入的值

    # In default setting, it will generate this autoencoder:
    # (input, 64, ReLU)
    # (64, 32, ReLU)
    # (32, 64, ReLU)
    # (64, output, Sigmoid)
    hidden_neurons = [X_train.shape[1] // 2, X_train.shape[1] // 4]

    classifiers = {
        'AE': AutoEncoder(hidden_neurons=hidden_neurons, epochs=epochs, preprocessing=False,
                          contamination=outliers_fraction,
                          learning_rate=learning_rate, batch_norm=True, dropout_rate=0.5, quadratic=False,
                          sub_learning_rate=alpha, batch_size=batch_size, hybird=False,
                          hybird_style='X'),
        'QAE': AutoEncoder(hidden_neurons=hidden_neurons, epochs=epochs, preprocessing=False,
                           contamination=outliers_fraction,
                           learning_rate=learning_rate, batch_norm=True, dropout_rate=0.5, quadratic=True,
                           sub_learning_rate=alpha, batch_size=batch_size, hybird=False,
                           hybird_style='X'),
        'HAE_X': AutoEncoder(hidden_neurons=hidden_neurons, epochs=epochs, preprocessing=False,
                             contamination=outliers_fraction,
                             learning_rate=learning_rate, batch_norm=True, dropout_rate=0.5, quadratic=True,
                             sub_learning_rate=alpha, batch_size=batch_size, hybird=True,
                             hybird_style='X'),
        'HAE_Y': AutoEncoder(hidden_neurons=hidden_neurons, epochs=epochs, preprocessing=False,
                             contamination=outliers_fraction,
                             learning_rate=learning_rate, batch_norm=True, dropout_rate=0.5, quadratic=True,
                             sub_learning_rate=alpha, batch_size=batch_size, hybird=True,
                             hybird_style='Y'),
        'HAE_I': AutoEncoder(hidden_neurons=hidden_neurons, epochs=epochs, preprocessing=False,
                             contamination=outliers_fraction,
                             learning_rate=learning_rate, batch_norm=True, dropout_rate=0.5, quadratic=True,
                             sub_learning_rate=alpha, batch_size=batch_size, hybird=True,
                             hybird_style='I')
    }

    clf = classifiers[ae_model]  # choose an autoencoder

    # standardizing data for processing
    X_train_norm, X_test_norm = standardizer(X_train, X_test)

    clf.fit(X_train_norm)
    test_scores = clf.decision_function(X_test_norm)
    y_predict,_ = clf.predict(X_test_norm)

    # we only use AUCs for comparison in the paper, however, the program will compute other metrics.
    auc = round(roc_auc_score(y_test, test_scores), ndigits=4)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_predict, average='macro')
    precision = round(precision, ndigits=4)
    recall = round(recall, ndigits=4)
    f1 = round(f1, ndigits=4)


    ae_result = [auc, precision, recall, f1]
    ae_result_np = np.array(ae_result).reshape(1, -1)
    temp_df = pd.DataFrame(ae_result_np, columns=df_columns)
    ae_result_df = pd.concat([ae_result_df, temp_df], axis=0)
    if not os.path.exists('results'):
        os.mkdir('results')
    ae_result_df.to_csv(
        'results/%s_%s_bs%d_lr%f_alpha_%f_seed%d.csv' % (mat_file[:-4], ae_model, batch_size, learning_rate,
                                                       alpha, seed), index=False)
