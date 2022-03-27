from pyod.utils import standardizer
from scipy.io import loadmat
from six.moves import cPickle as pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import torch as torch
from torch.autograd import Variable
from torch.utils.data import Dataset




def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di

def split_data(X, y):
    X_normal = X[np.where(y == 0)]
    y_normal = y[np.where(y == 0)]
    X_abnormal = X[np.where(y == 1)]
    y_abnormal = y[np.where(y == 1)]
    X_train, X_test, y_train, y_test = train_test_split(X_normal, y_normal, test_size=0.2)

    X_test = np.concatenate((X_test, X_abnormal))
    y_test = np.concatenate((y_test, y_abnormal))

    return X_train, X_test, y_train, y_test


class RealDataset(Dataset):
    def __init__(self, path, missing_ratio):


        # data = np.load(path, allow_pickle=True)
        data = loadmat(path)
        self.missing_ratio = missing_ratio
        self.x = data["X"]
        self.y = data["y"].ravel()
        n, d = self.x.shape
        mask = np.random.rand(n, d)
        mask = (mask > missing_ratio).astype(float)
        if missing_ratio > 0.0:
            self.x[mask == 0] = np.nan
            imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
            self.x = imputer.fit_transform(self.x)

        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.x, self.y)
        self.X_train, self.X_test = standardizer(self.X_train, self.X_test)

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
        if len(self.x.shape) > 2:
            raise Exception("only handles single channel data")
        else:
            return self.x.shape[1]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.array(self.X_train[idx, :])),
            torch.from_numpy(np.array(self.y_train[idx])),
            torch.from_numpy(np.array(self.X_test[idx, :])),
            torch.from_numpy(np.array(self.y_test[idx])),
        )

    def __sample__(self, num):
        len = self.__len__()
        index = np.random.choice(len, num, replace=False)
        return self.__getitem__(index)

    def __anomalyratio__(self):
        return self.y.sum() / self.y.shape[0]
