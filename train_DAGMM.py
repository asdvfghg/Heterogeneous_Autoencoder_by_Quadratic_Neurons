import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
import datetime
import torch.utils.data as data
from torch.autograd import grad
from torch.autograd import Variable
from Model.DAGMM import DaGMM
from utils.data_process import RealDataset
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm

'''
This implementation is based on https://github.com/danieltan07/dagmm and https://github.com/tnakae/DAGMM
We noticed that the training process is highly numerical unstable and two above implementation also mentioned that problem.
Specifically, we found when bottleneck dimension is high, the issue becomes severe. In the original paper of DAGMM,
the bottleneck dimension is 1 (without counting the cosine similarity and reconstruction loss). 
For example, If we increase it to 10, in many datasets, it will have numerical issue. 
Also, for unsupervised AD, it is very tricky to pick lambda, gmm_k, lambda_cov_diag, since there is no clean data to 
evaluate the performance.
'''
class Solver():
    DEFAULTS = {}

    def __init__(self, data_name, lambda_energy=0.1, lambda_cov_diag=0.005, hidden_dim=128, z_dim=10, seed=0, learning_rate=1e-3, gmm_k=2,
                 batch_size=128, training_ratio=0.8, validation_ratio=0.1, max_epochs=100, missing_ratio=0.0):
        # Data loader
        self.gmm_k = gmm_k
        self.lambda_energy = lambda_energy
        self.lambda_cov_diag = lambda_cov_diag
        # read data here
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        data_path = "./data/" + data_name
        self.model_save_path = "./trained_model/{}/{}/DAGMM/{}/".format(data_name, missing_ratio, seed)
        self.result_path = "./results/{}/{}/DAGMM/{}/".format(data_name, missing_ratio, seed)
        os.makedirs(self.model_save_path, exist_ok=True)

        self.learning_rate = learning_rate
        self.missing_ratio = missing_ratio
        self.dataset = RealDataset(data_path, missing_ratio=self.missing_ratio)
        self.seed = seed
        self.max_epochs = max_epochs


        self.data_path = data_path
        self.data_anomaly_ratio = self.dataset.__anomalyratio__()
        self.input_dim = self.dataset.__dim__()
        self.data_normaly_ratio = 1 - self.data_anomaly_ratio
        n_sample = self.dataset.__len__()
        self.n_train = int(n_sample * (training_ratio))
        # self.n_validation = int(n_sample * validation_ratio)
        self.n_test = n_sample - self.n_train
        print('|data dimension: {}|data noise ratio:{}'.format(self.dataset.__dim__(), self.data_anomaly_ratio))

        self.hidden_dim = self.input_dim // 2
        self.z_dim = self.input_dim // 4
        self.data_normaly_ratio = 1 - self.data_anomaly_ratio

        self.training_loader = data.DataLoader(
            self.dataset.X_train, batch_size=batch_size, shuffle=True
        )

        self.testing_loader = data.DataLoader(
            self.dataset.X_test, batch_size=1, shuffle=False
        )



        self.training_loader = data.DataLoader(self.dataset.X_train, batch_size=batch_size, shuffle=True)
        # self.validation_loader = data.DataLoader(validation_data, batch_size=self.n_validation, shuffle=False)
        self.testing_loader = data.DataLoader(self.dataset.X_test, batch_size=1, shuffle=False)
        self.build_model()
        self.print_network()

    def build_model(self):
        # Define model
        self.dagmm = DaGMM(input_dim=self.input_dim, hidden_dim=self.hidden_dim, z_dim=self.z_dim, n_gmm=self.gmm_k)
        # Optimizers
        self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.learning_rate)
        # Print networks
        self.print_network()

        if torch.cuda.is_available():
            self.dagmm.cuda()

    def print_network(self):
        num_params = 0
        for p in self.dagmm.parameters():
            num_params += p.numel()
        # print(name)
        # print(model)
        print("The number of parameters: {}".format(num_params))

    def reset_grad(self):
        self.dagmm.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def train(self):
        iters_per_epoch = len(self.training_loader)

        start = 0
        # Start training
        iter_ctr = 0
        start_time = time.time()
        min_val_loss = 1e+15

        for e in tqdm(range(start, self.max_epochs)):
            for i, input_data in enumerate(self.training_loader):
                iter_ctr += 1
                start_time = time.time()

                input_data = self.to_var(input_data)

                # training
                total_loss, sample_energy, recon_error, cov_diag = self.dagmm_step(input_data)
                # Logging
                loss = {}
                loss['total_loss'] = total_loss.data.item()
                loss['sample_energy'] = sample_energy.item()
                loss['recon_error'] = recon_error.item()
                loss['cov_diag'] = cov_diag.item()

                self.dagmm.eval()


    def dagmm_step(self, input_data, validation_flag=False):
        input_data = input_data.float()
        if not validation_flag:
            self.optimizer.zero_grad()
            self.dagmm.train()

            enc, dec, z, gamma = self.dagmm(input_data)
            if torch.isnan(z.sum()):
                for p in self.dagmm.parameters():
                    print(p)
                print("pause")
            total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_function(input_data, dec, z, gamma,
                                                                                        self.lambda_energy,
                                                                                        self.lambda_cov_diag)

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.dagmm.parameters(), 5)
            self.optimizer.step()

        else:
            self.dagmm.eval()
            enc, dec, z, gamma = self.dagmm(input_data)

            total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_function(input_data, dec, z, gamma,
                                                                                        self.lambda_energy,
                                                                                        self.lambda_cov_diag)

        return total_loss, sample_energy, recon_error, cov_diag

    def test(self):
        print("======================TEST MODE======================")
        # self.dagmm.load_stat
        # self.dagmm.load_state_dict(torch.load(self.model_save_path + 'parameter.pth'))
        self.dagmm.eval()
        # self.data_loader.dataset.mode = "train"

        # compute the parameter of density estimation by using training and validation set
        N = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0

        for it, input_data in enumerate(self.training_loader):

            input_data = self.to_var(input_data)
            input_data = input_data.float()
            enc, dec, z, gamma = self.dagmm(input_data)
            phi, mu, cov = self.dagmm.compute_gmm_params(z, gamma)

            batch_gamma_sum = torch.sum(gamma, dim=0)

            gamma_sum += batch_gamma_sum
            mu_sum += mu * batch_gamma_sum.unsqueeze(-1)  # keep sums of the numerator only
            cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)  # keep sums of the numerator only

            N += input_data.size(0)

        train_phi = gamma_sum / N
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        print("N:", N)
        print("phi :\n", train_phi)
        print("mu :\n", train_mu)
        print("cov :\n", train_cov)

        train_energy = []
        train_labels = []
        train_z = []
        for it, input_data in enumerate(self.training_loader):
            input_data = self.to_var(input_data)
            input_data = input_data.float()
            enc, dec, z, gamma = self.dagmm(input_data)
            sample_energy, cov_diag = self.dagmm.compute_energy(z, phi=train_phi, mu=train_mu, cov=train_cov,
                                                                size_average=False)

            train_energy.append(sample_energy.data.cpu().numpy())
            train_z.append(z.data.cpu().numpy())
            # train_labels.append(labels.numpy())


        train_energy = np.concatenate(train_energy, axis=0)
        train_z = np.concatenate(train_z, axis=0)
        # train_labels = np.concatenate(train_labels, axis=0)

        test_energy = []
        test_labels = []
        test_z = []
        for it, input_data in enumerate(self.testing_loader):
            input_data = self.to_var(input_data)
            input_data = input_data.float()
            enc, dec, z, gamma = self.dagmm(input_data)
            sample_energy, cov_diag = self.dagmm.compute_energy(z, size_average=False)
            test_energy.append(sample_energy.data.cpu().numpy())
            test_z.append(z.data.cpu().numpy())
            # test_labels.append(labels.numpy())
        train_labels = self.dataset.y_train
        test_labels = self.dataset.y_test
        test_energy = np.concatenate(test_energy, axis=0)
        test_z = np.concatenate(test_z, axis=0)
        # test_labels = np.concatenate(test_labels, axis=0)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        combined_labels = np.concatenate([train_labels, test_labels], axis=0)



        thresh = np.percentile(combined_energy, self.data_normaly_ratio * 100)
        print("Threshold :", thresh)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)


        from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(gt, test_energy)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = prf(gt, pred, average='binary')

        os.makedirs(self.result_path, exist_ok=True)
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} auc:{:0.3f}".format(
            accuracy, precision, recall, f_score, auc))
        return accuracy, precision, recall, f_score, auc


if __name__ == '__main__':

    df_columns = ['AUC', 'Pre', 'Recall', 'F1']
    ae_result_df = pd.DataFrame(columns=df_columns)
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
    data_name = mat_file_list[0]
    auc_list = []

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")


    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    DAGMM_Solver = Solver(data_name=data_name, seed=seed,
                          learning_rate=3e-4, gmm_k=2, missing_ratio=0.0,
                          batch_size=128, training_ratio=0.8, max_epochs=1000)

    DAGMM_Solver.train()
    accuracy, precision, recall, f_score, auc = DAGMM_Solver.test()
    print("Data {} finished".format(data_name))
    auc_list.append(auc)

    ae_result = [auc, precision, recall, f_score]
    ae_result_np = np.array(ae_result).reshape(1, -1)
    ae_result_df = pd.DataFrame(ae_result_np, columns=df_columns)
    if not os.path.exists('results'):
        os.mkdir('results')
    ae_result_df.to_csv('results/%s_%s_%d.csv' % (data_name[:-4], 'DAGMM',seed), index=False)