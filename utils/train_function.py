import math

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn.modules import loss
import torch.nn.functional as F

'''
train functions for quadratic models
'''


# region QAE
def group_parameters(m):
    group_r, group_g, group_b, group_others = [], [], [], []
    for name, p in m.named_parameters():
        if '_r' in name:
            group_r += [p]
        elif '_g' in name:
            group_g += [p]
        elif '_b' in name:
            group_b += [p]
        else:
            group_others += [p]

    return (group_r, group_g, group_b, group_others)




# endregion

def split_data(X, y, random_state, missing_ratio=0.0):
    n, d = X.shape
    mask = np.random.rand(n, d)
    mask = (mask > missing_ratio).astype(float)
    if missing_ratio > 0.0:
        X[mask == 0] = np.nan
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        X = imputer.fit_transform(X)


    X_normal = X[np.where(y == 0)]
    y_normal = y[np.where(y == 0)]
    X_abnormal = X[np.where(y == 1)]
    y_abnormal = y[np.where(y == 1)]

    X_train, X_test, y_train, y_test = train_test_split(X_normal, y_normal, test_size=0.2,
                                                        random_state=random_state)

    X_test = np.concatenate([X_test, X_abnormal])
    y_test = np.concatenate([y_test, y_abnormal])




    return X_train, X_test, y_train, y_test
