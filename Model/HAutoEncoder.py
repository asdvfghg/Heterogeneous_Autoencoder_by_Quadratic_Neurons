from __future__ import division
from __future__ import print_function

# -*- coding: utf-8 -*-
from fvcore.nn import flop_count_str, FlopCountAnalysis

"""Using AutoEncoder with Outlier Detection (PyTorch)
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

'''
heterogeneous autoencoder, the original version follows pyod
'''

import torch
from torch import nn

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from utils.QuadraticOperation import QuadraticOperation
from utils.train_function import group_parameters
from pyod.models.base import BaseDetector
from pyod.utils.torch_utility import get_activation_by_name
from pyod.utils.stat_models import pairwise_distances_no_broadcast


class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader
    """

    def __init__(self, X, y=None, mean=None, std=None):
        super(PyODDataset, self).__init__()
        self.X = X
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]

        if self.mean is not None:
            sample = (sample - self.mean) / self.std

        return torch.from_numpy(sample), idx


class inner_autoencoder(nn.Module):
    def __init__(self,
                 n_features,
                 hidden_neurons=[128, 64],
                 dropout_rate=0.5,
                 batch_norm=True,
                 hidden_activation='relu',
                 quadratic=False):
        super(inner_autoencoder, self).__init__()
        self.n_features = n_features
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.hidden_activation = hidden_activation

        self.activation = get_activation_by_name(hidden_activation)

        self.layers_neurons_ = [self.n_features, *hidden_neurons]
        self.layers_neurons_decoder_ = self.layers_neurons_[::-1]
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.quadratic = quadratic

        for idx, layer in enumerate(self.layers_neurons_[:-1]):
            if batch_norm:
                self.encoder.add_module("batch_norm" + str(idx),
                                        nn.BatchNorm1d(
                                            self.layers_neurons_[idx]))
            if self.quadratic:
                self.encoder.add_module("quadratic linear" + str(idx),
                                        QuadraticOperation(self.layers_neurons_[idx],
                                                           self.layers_neurons_[
                                                               idx + 1]))
            else:
                self.encoder.add_module("linear" + str(idx),
                                        torch.nn.Linear(self.layers_neurons_[idx],
                                                        self.layers_neurons_[
                                                            idx + 1]))
            self.encoder.add_module(self.hidden_activation + str(idx),
                                    self.activation)
            self.encoder.add_module("dropout" + str(idx),
                                    torch.nn.Dropout(dropout_rate))

        for idx, layer in enumerate(self.layers_neurons_[:-2]):
            if batch_norm:
                self.decoder.add_module("batch_norm" + str(idx),
                                        nn.BatchNorm1d(
                                            self.layers_neurons_decoder_[idx]))

            if self.quadratic:
                self.decoder.add_module("quadratic linear" + str(idx),
                                        QuadraticOperation(self.layers_neurons_decoder_[idx],
                                                           self.layers_neurons_decoder_[
                                                               idx + 1]))
            else:
                self.decoder.add_module("linear" + str(idx),
                                        torch.nn.Linear(self.layers_neurons_decoder_[idx],
                                                        self.layers_neurons_decoder_[
                                                            idx + 1]))

            self.decoder.add_module(self.hidden_activation + str(idx),
                                    self.activation)

            self.decoder.add_module("dropout" + str(idx),
                                    torch.nn.Dropout(dropout_rate))
        if self.quadratic:
            self.decoder.add_module("quadratic linear" + str(-1),
                                    QuadraticOperation(self.layers_neurons_decoder_[-2],
                                                       self.layers_neurons_decoder_[
                                                           -1]))
        else:
            self.decoder.add_module("linear" + str(-1),
                                    torch.nn.Linear(self.layers_neurons_decoder_[-2],
                                                    self.layers_neurons_decoder_[
                                                        -1]))
        self.decoder.add_module("sigmoid", get_activation_by_name('sigmoid'))
        self.decoder.add_module("dropout" + str(-1),
                                torch.nn.Dropout(dropout_rate))

    def forward(self, x):
        # we could return the latent representation here after the encoder as the latent representation
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# class heterogeneous_autoencoder(nn.Module):
#     def __init__(self,
#                  n_features,
#                  hidden_neurons=[128, 64],
#                  dropout_rate=0.5,
#                  batch_norm=True,
#                  hidden_activation='relu',
#                  hybird_style='X'
#                  ):
#         super(heterogeneous_autoencoder, self).__init__()
#         self.n_features = n_features
#         self.dropout_rate = dropout_rate
#         self.batch_norm = batch_norm
#         self.hidden_activation = hidden_activation
#         self.hybird_style = hybird_style
#         self.activation = get_activation_by_name(hidden_activation)
#
#         self.layers_neurons_ = [self.n_features, *hidden_neurons]
#         self.layers_neurons_decoder_ = self.layers_neurons_[::-1]
#         # quadratic
#         self.encoderq = nn.Sequential()
#         self.decoderq = nn.Sequential()
#         # conventional
#         self.encoderc = nn.Sequential()
#         self.decoderc = nn.Sequential()
#         # I style
#         self.encoderi = nn.Sequential()
#         self.decoderi = nn.Sequential()
#
#         self.sig = nn.Sigmoid()
#         if self.hybird_style in ('X', 'Y'):
#             # region qudratic layer
#             for idx, layer in enumerate(self.layers_neurons_[:-1]):
#                 if batch_norm:
#                     self.encoderq.add_module("batch_norm" + str(idx),
#                                              nn.BatchNorm1d(
#                                                  self.layers_neurons_[idx]))
#                 self.encoderq.add_module("quadratic linear" + str(idx),
#                                          QuadraticOperation(self.layers_neurons_[idx],
#                                                             self.layers_neurons_[
#                                                                 idx + 1]))
#
#                 self.encoderq.add_module(self.hidden_activation + str(idx),
#                                          self.activation)
#                 self.encoderq.add_module("dropout" + str(idx),
#                                          torch.nn.Dropout(dropout_rate))
#
#             for idx, layer in enumerate(self.layers_neurons_[:-2]):
#                 if batch_norm:
#                     self.decoderq.add_module("batch_norm" + str(idx),
#                                              nn.BatchNorm1d(
#                                                  self.layers_neurons_decoder_[idx]))
#
#                 self.decoderq.add_module("quadratic linear" + str(idx),
#                                          QuadraticOperation(self.layers_neurons_decoder_[idx],
#                                                             self.layers_neurons_decoder_[
#                                                                 idx + 1]))
#
#                 self.decoderq.add_module(self.hidden_activation + str(idx),
#                                          self.activation)
#
#                 self.decoderq.add_module("dropout" + str(idx),
#                                          torch.nn.Dropout(dropout_rate))
#
#             self.decoderq.add_module("batch_norm" + '-1', torch.nn.BatchNorm1d(self.layers_neurons_decoder_[-2]))
#             self.decoderq.add_module("quadratic linear" + str(-1),
#                                      torch.nn.Linear(self.layers_neurons_decoder_[-2],
#                                                      self.layers_neurons_decoder_[
#                                                          -1]))
#             # endregion
#
#             # region conventional layer
#             for idx, layer in enumerate(self.layers_neurons_[:-1]):
#                 if batch_norm:
#                     self.encoderc.add_module("batch_norm" + str(idx),
#                                              nn.BatchNorm1d(
#                                                  self.layers_neurons_[idx]))
#                 self.encoderc.add_module("linear" + str(idx),
#                                         torch.nn.Linear(self.layers_neurons_[idx],
#                                                         self.layers_neurons_[
#                                                             idx + 1]))
#
#                 self.encoderc.add_module(self.hidden_activation + str(idx),
#                                          self.activation)
#                 self.encoderc.add_module("dropout" + str(idx),
#                                          torch.nn.Dropout(dropout_rate))
#
#             for idx, layer in enumerate(self.layers_neurons_[:-2]):
#                 if batch_norm:
#                     self.decoderc.add_module("batch_norm" + str(idx),
#                                              nn.BatchNorm1d(
#                                                  self.layers_neurons_decoder_[idx]))
#
#                 self.decoderc.add_module("linear" + str(idx),
#                                         torch.nn.Linear(self.layers_neurons_decoder_[idx],
#                                                         self.layers_neurons_decoder_[
#                                                             idx + 1]))
#
#                 self.decoderc.add_module(self.hidden_activation + str(idx),
#                                          self.activation)
#
#                 self.decoderc.add_module("dropout" + str(idx),
#                                          torch.nn.Dropout(dropout_rate))
#
#             self.decoderc.add_module("batch_norm" + '-1', torch.nn.BatchNorm1d(self.layers_neurons_decoder_[-2]))
#             self.decoderc.add_module("linear" + str(-1),
#                                      torch.nn.Linear(self.layers_neurons_decoder_[-2],
#                                                      self.layers_neurons_decoder_[
#                                                          -1]))
#             # endregion
#
#         # region Style 'I'
#         if self.hybird_style == 'I':
#             for idx, layer in enumerate(self.layers_neurons_[:-1]):
#                 # even layer, set to quadratic operation
#                 if idx % 2 == 0:
#                     if batch_norm:
#                         self.encoderi.add_module("batch_norm" + str(idx),
#                                                  nn.BatchNorm1d(
#                                                      self.layers_neurons_[idx]))
#                     self.encoderi.add_module("quadratic linear" + str(idx),
#                                              QuadraticOperation(self.layers_neurons_[idx],
#                                                                 self.layers_neurons_[
#                                                                     idx + 1]))
#
#                     self.encoderi.add_module(self.hidden_activation + str(idx),
#                                              self.activation)
#                     self.encoderi.add_module("dropout" + str(idx),
#                                              torch.nn.Dropout(dropout_rate))
#                 # odd layer, set to linear operation
#                 if idx % 2 == 1:
#                     if batch_norm:
#                         self.encoderi.add_module("batch_norm" + str(idx),
#                                                  nn.BatchNorm1d(
#                                                      self.layers_neurons_[idx]))
#                     self.encoderi.add_module("linear" + str(idx),
#                                             torch.nn.Linear(self.layers_neurons_[idx],
#                                                             self.layers_neurons_[
#                                                                 idx + 1]))
#
#                     self.encoderi.add_module(self.hidden_activation + str(idx),
#                                              self.activation)
#                     self.encoderi.add_module("dropout" + str(idx),
#                                              torch.nn.Dropout(dropout_rate))
#                 # if idx + 3 == len(self.layers_neurons_):
#                 #     break
#             for idx, layer in enumerate(self.layers_neurons_[:-2]):
#                 #  even layer, set to linear operation
#                 if idx % 2 == 0:
#                     if batch_norm:
#                         self.decoderi.add_module("batch_norm" + str(idx),
#                                                  nn.BatchNorm1d(
#                                                      self.layers_neurons_decoder_[idx]))
#
#                     self.decoderi.add_module("linear" + str(idx),
#                                             torch.nn.Linear(self.layers_neurons_decoder_[idx],
#                                                             self.layers_neurons_decoder_[
#                                                                 idx + 1]))
#
#                     self.decoderi.add_module(self.hidden_activation + str(idx),
#                                              self.activation)
#
#                     self.decoderi.add_module("dropout" + str(idx),
#                                              torch.nn.Dropout(dropout_rate))
#
#                 #  odd layer, set to quadratic operation
#                 if idx % 2 == 1:
#                     if batch_norm:
#                         self.decoderi.add_module("batch_norm" + str(idx),
#                                                  nn.BatchNorm1d(
#                                                      self.layers_neurons_decoder_[idx]))
#
#                     self.decoderi.add_module("quadratic linear" + str(idx),
#                                              QuadraticOperation(self.layers_neurons_decoder_[idx],
#                                                                 self.layers_neurons_decoder_[
#                                                                     idx + 1]))
#
#                     self.decoderi.add_module(self.hidden_activation + str(idx+1),
#                                              self.activation)
#
#                     self.decoderi.add_module("dropout" + str(idx+1),
#                                              torch.nn.Dropout(dropout_rate))
#
#         # endregion
#         # self.decoderq.add_module("sigmoid", get_activation_by_name('sigmoid'))
#
#     def forward(self, x):
#         if self.hybird_style == 'X':
#             z1 = self.encoderq(x)
#             z2 = self.encoderc(x)
#             encoded = z1 + z2
#             d1 = self.decoderq(encoded)
#             d2 = self.decoderc(encoded)
#             y = d1 + d2
#             decoded = self.sig(y)
#             # drop = torch.nn.Dropout(0.5)
#             # decoded = drop(decoded)
#         if self.hybird_style == 'Y':
#             z1 = self.encoderq(x)
#             z2 = self.encoderc(x)
#             encoded = z1 + z2
#             y = self.decoderq(encoded)
#             decoded = self.sig(y)
#
#         if self.hybird_style == 'I':
#             encoded = self.encoderi(x)
#             y = self.decoderi(encoded)
#             decoded = self.sig(y)
#         return encoded, decoded


class heterogeneous_autoencoder(nn.Module):
    def __init__(self,
                 n_features,
                 hidden_neurons=[128, 64],
                 dropout_rate=0.5,
                 hybird_style='X'):
        super(heterogeneous_autoencoder, self).__init__()
        self.hybird_style = hybird_style
        self.bn1 = nn.BatchNorm1d(n_features)
        self.encq1 = QuadraticOperation(n_features, hidden_neurons[0])
        self.enc1 = nn.Linear(n_features, hidden_neurons[0])
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dp11 = torch.nn.Dropout(dropout_rate)
        self.dp12 = torch.nn.Dropout(dropout_rate)

        self.bn21 = nn.BatchNorm1d(hidden_neurons[0])
        self.bn22 = nn.BatchNorm1d(hidden_neurons[0])
        self.encq2 = QuadraticOperation(hidden_neurons[0], hidden_neurons[1])
        self.enc2 = nn.Linear(hidden_neurons[0], hidden_neurons[1])
        self.relu3 = nn.ReLU()
        self.dp2 = torch.nn.Dropout(dropout_rate)

        self.bn3 = nn.BatchNorm1d(hidden_neurons[1])
        self.decq1 = QuadraticOperation(hidden_neurons[1], hidden_neurons[0])
        self.dec1 = nn.Linear(hidden_neurons[1], hidden_neurons[0])
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.dp31 = torch.nn.Dropout(dropout_rate)
        self.dp32 = torch.nn.Dropout(dropout_rate)

        self.bn41 = nn.BatchNorm1d(hidden_neurons[0])
        self.bn42 = nn.BatchNorm1d(hidden_neurons[0])
        self.decq2 = QuadraticOperation(hidden_neurons[0], n_features)
        self.dec2 = nn.Linear(hidden_neurons[0], n_features)
        self.sig = nn.Sigmoid()
        self.dp4 = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.hybird_style == 'X':
            x = self.bn1(x)
            y11 = self.encq1(x)
            y12 = self.enc1(x)
            y11 = self.relu1(y11)
            y12 = self.relu2(y12)
            y11 = self.dp11(y11)
            y12 = self.dp12(y12)

            y11 = self.bn21(y11)
            y12 = self.bn22(y12)
            y21 = self.encq2(y11)
            y22 = self.enc2(y12)
            y2 = y21 + y22
            encoded = self.relu3(y2)
            encoded = self.dp2(encoded)
            encoded = self.bn3(encoded)

            y31 = self.decq1(encoded)
            y32 = self.dec1(encoded)
            y31 = self.relu4(y31)
            y32 = self.relu5(y32)
            y31 = self.dp31(y31)
            y32 = self.dp32(y32)
            y31 = self.bn41(y31)
            y32 = self.bn42(y32)
            y41 = self.decq2(y31)
            y42 = self.dec2(y32)
            y4 = y41 + y42
            decoded = self.sig(y4)
            decoded = self.dp4(decoded)

        if self.hybird_style == 'Y':
            x = self.bn1(x)
            y11 = self.encq1(x)
            y12 = self.enc1(x)
            y11 = self.relu1(y11)
            y12 = self.relu2(y12)
            y11 = self.dp11(y11)
            y12 = self.dp12(y12)
            y11 = self.bn21(y11)
            y12 = self.bn22(y12)
            y21 = self.encq2(y11)
            y22 = self.enc2(y12)
            y2 = y21 + y22
            encoded = self.relu3(y2)
            encoded = self.dp2(encoded)
            encoded = self.bn3(encoded)

            y31 = self.decq1(encoded)
            y31 = self.relu4(y31)
            y31 = self.dp31(y31)
            y31 = self.bn41(y31)
            y41 = self.decq2(y31)
            decoded = self.sig(y41)
            decoded = self.dp4(decoded)

        # region I
        if self.hybird_style == 'I':
            x = self.bn1(x)
            y11 = self.encq1(x)
            y11 = self.relu1(y11)
            y11 = self.dp11(y11)
            y11 = self.bn21(y11)
            y2 = self.enc2(y11)
            encoded = self.relu3(y2)
            encoded = self.dp2(encoded)

            encoded = self.bn3(encoded)
            y32 = self.dec1(encoded)
            y32 = self.relu5(y32)
            y32 = self.dp32(y32)
            y32 = self.bn42(y32)
            y41 = self.decq2(y32)
            decoded = self.sig(y41)
            decoded = self.dp4(decoded)
        # endregion

        return decoded


class AutoEncoder(BaseDetector):
    """Auto Encoder (AE) is a type of neural networks for learning useful data
    representations in an unsupervised manner. Similar to PCA, AE could be used
    to detect outlying objects in the data by calculating the reconstruction
    errors. See :cite:`aggarwal2015outlier` Chapter 3 for details.

    Notes
    -----
        This is the PyTorch version of AutoEncoder. See auto_encoder.py for
        the TensorFlow version.

        The documentation is not finished!

    Parameters
    ----------
    hidden_neurons : list, optional (default=[64, 32])
        The number of neurons per hidden layers. So the network has the
        structure as [n_features, 64, 32, 32, 64, n_features]

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.
        See https://keras.io/activations/

    batch_norm : boolean, optional (default=True)
        Whether to apply Batch Normalization,
        See https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

    loss : str or obj, optional (default=torch.nn.MSELoss)
        String (name of objective function) or objective function.
        NOT SUPPORT FOR CHANGE YET.

    optimizer : str, optional (default='adam')
        String (name of optimizer) or optimizer instance.
        NOT SUPPORT FOR CHANGE YET.

    epochs : int, optional (default=100)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    l2_regularizer : float in (0., 1), optional (default=0.1)
        The regularization strength of activity_regularizer
        applied on each layer. By default, l2 regularizer is used. See
        https://keras.io/regularizers/

    validation_size : float in (0., 1), optional (default=0.1)
        The percentage of data to be used for validation.

    preprocessing : bool, optional (default=True)
        If True, apply standardization on the data.

    verbose : int, optional (default=1)
        Verbosity mode.

        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.

        For verbose >= 1, model summary may be printed.

    random_state : random_state: int, RandomState instance or None, optional
        (default=None)
        If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. When fitting this is used
        to define the threshold on the decision function.

    Attributes
    ----------
    encoding_dim_ : int
        The number of neurons in the encoding layer.

    compression_rate_ : float
        The ratio between the original feature and
        the number of neurons in the encoding layer.

    model_ : Keras Object
        The underlying AutoEncoder in Keras.

    history_: Keras Object
        The AutoEncoder training history.

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self,
                 hidden_neurons=None,
                 hidden_activation='relu',
                 batch_norm=True,
                 # loss='mse',
                 # optimizer='adam',
                 learning_rate=1e-3,
                 sub_learning_rate=0.1,
                 epochs=100,
                 batch_size=32,
                 dropout_rate=0.2,
                 # l2_regularizer=0.1,
                 weight_decay=1e-5,
                 # validation_size=0.1,
                 preprocessing=True,
                 loss_fn=None,
                 # verbose=1,
                 # random_state=None,
                 contamination=0.1,
                 device=None,
                 quadratic=False,
                 hybird=False,
                 hybird_style='X'):
        super(AutoEncoder, self).__init__(contamination=contamination)
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate

        self.epochs = epochs
        self.batch_size = batch_size

        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.preprocessing = preprocessing
        self.sub_learning_rate = sub_learning_rate
        self.hybird_sytle = hybird_style
        if loss_fn is None:
            self.loss_fn = torch.nn.MSELoss()

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # default values
        if self.hidden_neurons is None:
            self.hidden_neurons = [64, 32]

        # self.verbose = verbose
        self.quadratic = quadratic
        self.hybird = hybird

    # noinspection PyUnresolvedReferences
    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        n_samples, n_features = X.shape[0], X.shape[1]

        # conduct standardization if needed
        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.mean(X, axis=0)
            train_set = PyODDataset(X=X, mean=self.mean, std=self.std)

        else:
            train_set = PyODDataset(X=X)

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)

        # initialize the model
        if self.hybird == False:
            self.model = inner_autoencoder(
                n_features=n_features,
                hidden_neurons=self.hidden_neurons,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                hidden_activation=self.hidden_activation,
                quadratic=self.quadratic)
        if self.hybird == True:
            self.model = heterogeneous_autoencoder(
                n_features=n_features,
                hidden_neurons=self.hidden_neurons,
                dropout_rate=self.dropout_rate,
                hybird_style=self.hybird_sytle
            )
        # move to device and print model information
        self.model = self.model.to(self.device)
        print(self.model)

        # train the autoencoder to find the best one
        self._train_autoencoder(train_loader)

        self.model.load_state_dict(self.best_model_dict)
        self.decision_scores_ = self.decision_function(X)

        self._process_decision_scores()
        return self

    def predict(self, X):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        pred_score = self.decision_function(X)

        latent_np = np.ones([1, self.hidden_neurons[-1]])
        for l in self.latent_list:
            latent_np = np.vstack([latent_np, l])
        latent_np = latent_np[1:, ]

        return (pred_score > self.threshold_).astype('int').ravel(), latent_np

    def _train_autoencoder(self, train_loader):
        """Internal function to train the autoencoder

        Parameters
        ----------
        train_loader : torch dataloader
            Train data.
        """
        if self.quadratic:
            group_r, group_g, group_b = group_parameters(self.model)
            sub_rate = self.sub_learning_rate * self.learning_rate
            optimizer = torch.optim.Adam([
                {"params": group_r},
                {"params": group_g, "lr": sub_rate},
                {"params": group_b, "lr": sub_rate},
            ], lr=self.learning_rate, weight_decay=self.weight_decay)

        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate,
                weight_decay=self.weight_decay)

        self.best_loss = float('inf')
        self.best_model_dict = None

        for epoch in range(self.epochs):
            overall_loss = []
            for data, data_idx in train_loader:
                data = data.to(self.device).float()
                _, dat = self.model(data)
                loss = self.loss_fn(data, dat)

                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                overall_loss.append(loss.item())
            print('epoch {epoch}: training loss {train_loss} '.format(
                epoch=epoch, train_loss=np.mean(overall_loss)))

            # track the best model so far
            if np.mean(overall_loss) <= self.best_loss:
                # print("epoch {ep} is the current best; loss={loss}".format(ep=epoch, loss=np.mean(overall_loss)))
                self.best_loss = np.mean(overall_loss)
                self.best_model_dict = self.model.state_dict()

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model', 'best_model_dict'])
        X = check_array(X)

        # note the shuffle may be true but should be False
        if self.preprocessing:
            dataset = PyODDataset(X=X, mean=self.mean, std=self.std)
        else:
            dataset = PyODDataset(X=X)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False)
        # enable the evaluation mode
        self.model.eval()
        self.latent_list = []
        # construct the vector for holding the reconstruction error
        outlier_scores = np.zeros([X.shape[0], ])
        with torch.no_grad():
            for i, (data, idx) in enumerate(dataloader):
                data_cuda = data.to(self.device).float()
                latent, predict = self.model(data_cuda)
                predict = predict.cpu().numpy()
                latent = latent.cpu().numpy()
                self.latent_list.append(latent)
                # this is the outlier score
                score = pairwise_distances_no_broadcast(
                    data, predict)
                if np.any(np.isnan(score)) or np.any(np.isinf(score)):
                    score = 10000.0
                outlier_scores[idx] = score

        return outlier_scores

if __name__ == "__main__":
    X = torch.randn((10, 274))
    hidden_neurons = [X.shape[1]*2, X.shape[1] // 4]
    model = heterogeneous_autoencoder(
        n_features=X.shape[1],
        hidden_neurons=hidden_neurons,
        hybird_style='X'
    )
    print(flop_count_str(FlopCountAnalysis(model, X)))