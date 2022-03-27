from __future__ import division
from __future__ import print_function

import os
import sys

from train_ae import random_seed
from utils.train_function import split_data
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.suod import SUOD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.deep_svdd import DeepSVDD
from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
if __name__ == "__main__":

    # Define data file and read X and y
    mat_file_list = [
         'arrhythmia.mat',
        'glass.mat',
        'musk.mat',
        'optdigits.mat',
        'pendigits.mat',
        'pima.mat',
        'vertebral.mat',
        'wbc.mat',
    ]

    random_state = np.random.seed(42)  # Numpy module.

    detector_list = [LOF(n_neighbors=15), LOF(n_neighbors=20),
                     LOF(n_neighbors=25), LOF(n_neighbors=35)]

    df_columns = ['Data', '#Samples', '# Dimensions', 'Outlier Perc',
                'LOF', 'OCSVM', 'SUOD', 'SO-GAAL', 'DeepSVD']
    auc_df = pd.DataFrame(columns=df_columns)
    prn_df = pd.DataFrame(columns=df_columns)
    pre_df = pd.DataFrame(columns=df_columns)
    recall_df = pd.DataFrame(columns=df_columns)
    f1_df = pd.DataFrame(columns=df_columns)

    for mat_file in mat_file_list:
        print("\n... Processing", mat_file, '...')
        mat = loadmat(os.path.join('.//data', mat_file))

        X = mat['X']
        y = mat['y'].ravel()

        outliers_fraction = np.count_nonzero(y) / len(y)
        outliers_percentage = round(outliers_fraction * 100, ndigits=4)

        # construct containers for saving results [‘wbc’, 378, 30, 5.5556]
        auc_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
        prn_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
        pre_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
        recall_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
        f1_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
        time_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]


        X_train, X_test, y_train, y_test = split_data(X, y, random_state)
        # standardizing data for processing
        X_train_norm, X_test_norm = standardizer(X_train, X_test)

        classifiers = {'Local Outlier Factor (LOF)': LOF(
                contamination=outliers_fraction),
                       'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
                       'SUOD': SUOD(base_estimators=detector_list, n_jobs=2, combination='average', contamination=outliers_fraction,
               verbose=False),
                       'SO-GAAL': SO_GAAL(contamination=outliers_fraction),
                       'DeepSVDD': DeepSVDD(epochs=200, contamination=outliers_fraction, random_state=random_state)
        }



        for clf_name, clf in classifiers.items():
            print('Dataset:' + mat_file +' '+ 'Classifier:' + clf_name)
            clf.fit(X_train_norm)
            test_scores = clf.decision_function(X_test_norm)
            y_predict = clf.predict(X_test_norm)


            auc = round(roc_auc_score(y_test, test_scores), ndigits=4)
            prn = round(precision_n_scores(y_test, test_scores), ndigits=4)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_predict, average='binary')
            precision = round(precision, ndigits=4)
            recall = round(recall, ndigits=4)
            f1 = round(f1, ndigits=4)

            auc_list.append(auc)
            prn_list.append(prn)
            pre_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)


        temp_df = pd.DataFrame(auc_list).transpose()
        temp_df.columns = df_columns
        auc_df = pd.concat([auc_df, temp_df], axis=0)

        temp_df = pd.DataFrame(prn_list).transpose()
        temp_df.columns = df_columns
        prn_df = pd.concat([prn_df, temp_df], axis=0)

        temp_df = pd.DataFrame(pre_list).transpose()
        temp_df.columns = df_columns
        pre_df = pd.concat([pre_df, temp_df], axis=0)

        temp_df = pd.DataFrame(recall_list).transpose()
        temp_df.columns = df_columns
        recall_df = pd.concat([recall_df, temp_df], axis=0)

        temp_df = pd.DataFrame(f1_list).transpose()
        temp_df.columns = df_columns
        f1_df = pd.concat([f1_df, temp_df], axis=0)

        if not os.path.exists('results'):
            os.mkdir('results')
        auc_df.to_csv('results/benchmark_auc.csv', index=False)
        prn_df.to_csv('results/benchmark_prn.csv', index=False)
        pre_df.to_csv('results/benchmark_pre.csv', index=False)
        recall_df.to_csv('results/benchmark_recall.csv', index=False)
        f1_df.to_csv('results/benchmark_f1.csv', index=False)