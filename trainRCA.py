import pandas as pd
import torch as torch
import os

import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import argparse

from Model.RCA import AE
from utils.data_process import RealDataset


class Solver_RCA:
    def __init__(
        self,
        data_name,

        seed=0,  # random seed
        learning_rate=1e-3,  # learning rate
        batch_size=128,  #  batchsize
        training_ratio=0.8,  #  training data percentage
        max_epochs=100,  #  training epochs
        coteaching=1.0,  #  whether selects sample based on loss value
        oe=0.0,  # how much we overestimate the ground-truth anomaly ratio
        missing_ratio=0.0,  # missing ratio in the data
    ):
        # Data loader
        # read data here
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        use_cuda = torch.cuda.is_available()
        self.data_name = data_name
        self.device = torch.device("cuda" if use_cuda else "cpu")
        data_path = "./data/" + data_name
        self.missing_ratio = missing_ratio
        self.model_save_path = "./trained_model/{}/{}/RCA/{}/".format(
            data_name, missing_ratio, seed
        )
        if oe == 0.0:
            self.result_path = "./results/{}/{}/RCA/{}/".format(
                data_name, missing_ratio, seed
            )
        else:
            self.result_path = "./results/{}/{}/RCA_{}/{}/".format(
                data_name, missing_ratio, oe, seed
            )

        os.makedirs(self.model_save_path, exist_ok=True)
        self.learning_rate = learning_rate
        self.dataset = RealDataset(
            data_path, missing_ratio=self.missing_ratio
        )
        self.seed = seed

        self.max_epochs = max_epochs
        self.coteaching = coteaching
        self.beta = 0.0  # initially, select all data
        self.alpha = 0.5
        self.data_path = data_path

        self.data_anomaly_ratio = self.dataset.__anomalyratio__() + oe

        self.input_dim = self.dataset.__dim__()
        self.hidden_dim = self.input_dim // 2
        self.z_dim = self.input_dim // 4
        self.data_normaly_ratio = 1 - self.data_anomaly_ratio

        n_sample = self.dataset.__len__()
        self.n_train = int(n_sample * (training_ratio))
        self.n_test = n_sample - self.n_train
        print(
            "|data dimension: {}|data noise ratio:{}".format(
                self.dataset.__dim__(), self.data_anomaly_ratio
            )
        )

        self.decay_ratio = abs(self.beta - (1 - self.data_anomaly_ratio)) / (
            self.max_epochs / 2
        )



        # training_data, testing_data = data.random_split(
        #     dataset=self.dataset, lengths=[self.n_train, self.n_test]
        # )

        self.training_loader = data.DataLoader(
            self.dataset.X_train, batch_size=batch_size, shuffle=True
        )

        self.testing_loader = data.DataLoader(
            self.dataset.X_test, batch_size=1, shuffle=False
        )
        self.ae = None
        self.discriminator = None
        self.build_model()
        self.print_network()

    def build_model(self):
        self.ae = AE(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim, z_dim=self.z_dim
        )
        self.ae = self.ae.to(self.device)

    def print_network(self):
        num_params = 0
        for p in self.ae.parameters():
            num_params += p.numel()
        print("The number of parameters: {}".format(num_params))

    def train(self):
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=self.learning_rate)
        self.ae.eval()
        loss_mse = torch.nn.MSELoss(reduction='none')
        if self.data_name == 'optdigits':
            loss_mse = torch.nn.BCELoss(reduction='none')

        for epoch in tqdm(range(self.max_epochs)):  # train 3 time classifier
            for i, x in enumerate(self.training_loader):
                x = x.to(self.device).float()
                n = x.shape[0]
                n_selected = int(n * (1-self.beta))

                # if config.coteaching == 0.0:
                #     n_selected = n
                if i == 0:
                    current_ratio = "{}/{}".format(n_selected, n)

                optimizer.zero_grad()

                with torch.no_grad():
                    self.ae.eval()
                    z1, z2, xhat1, xhat2 = self.ae(x.float(), x.float())

                    error1 = loss_mse(xhat1, x)
                    error1 = error1
                    error2 = loss_mse(xhat2, x)
                    error2 = error2

                    error1 = error1.sum(dim=1)
                    error2 = error2.sum(dim=1)
                    _, index1 = torch.sort(error1)
                    _, index2 = torch.sort(error2)

                    index1 = index1[:n_selected]
                    index2 = index2[:n_selected]

                    x1 = x[index2, :]
                    x2 = x[index1, :]


                self.ae.train()
                z1, z2, xhat1, xhat2 = self.ae(x1.float(), x2.float())
                loss = loss_mse(xhat1, x1) + loss_mse(xhat2, x2)
                loss = loss.sum()
                loss.backward()
                optimizer.step()

            if self.beta < self.data_anomaly_ratio:
                self.beta = min(
                    self.data_anomaly_ratio, self.beta + self.decay_ratio
                )

    def test(self):
        print("======================TEST MODE======================")
        self.ae.train()
        mse_loss = torch.nn.MSELoss(reduction='none')
        if self.data_name == 'optdigits':
            mse_loss = torch.nn.BCELoss(reduction='none')

        error_list = []
        with torch.no_grad():
            for _, x in enumerate(self.testing_loader):  # testing data loader has n_test batchsize, if it is image data, need change this part
                x = x.to(self.device).float()
                _, _, xhat1, xhat2 = self.ae(x.float(), x.float())
                error = mse_loss(xhat1, x) + mse_loss(xhat2, x)
                error = error.mean(dim=1)
                error = error.data.cpu().numpy()
                error_list.append(error)
        error_list = np.array(error_list).reshape(-1)
        # error = error_list.mean(axis=0)
        from sklearn.metrics import (
            precision_recall_fscore_support as prf,
            accuracy_score,
            roc_auc_score,
        )
        y = self.dataset.y_test
        thresh = np.percentile(error_list, self.dataset.__anomalyratio__() * 100)
        print("Threshold :", thresh)
        #
        pred = (error_list > thresh).astype(int)
        gt = y.astype(int)
        gt = gt.ravel()
        auc = roc_auc_score(gt, error_list)
        precision, recall, f_score, support = prf(gt, pred, average="binary")

        print(
            "AUC : {:0.4f}".format(
                auc
            )
        )

        # os.makedirs(self.result_path, exist_ok=True)
        #
        # results_df = pd.DataFrame([accuracy], columns=['acc'])
        # results_df.to_csv(self.result_path + "result.csv", index=False)
        # np.save(
        #     self.result_path + "result.csv",
        #     {
        #         "accuracy": accuracy,
        #         "precision": precision,
        #         "recall": recall,
        #         "f1": f_score,
        #         "auc": auc,
        #     },
        # )
        print("result save to {}".format(self.result_path))
        return auc, precision, recall, f_score


if __name__ == "__main__":
    df_columns = ['AUC', 'Pre', 'Recall', 'F1']
    ae_result_df = pd.DataFrame(columns=df_columns)
    mat_file_list = [
        'arrhythmia.mat',
        # 'glass.mat',
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
    """
    read data
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    Solver = Solver_RCA(
        data_name=data_name,
        seed=seed,
        learning_rate=3e-4,
        batch_size=128,
        training_ratio=0.8,
        max_epochs=100,
        missing_ratio=0.0,
        oe=0.0,
    )

    Solver.train()
    auc, precision, recall, f_score = Solver.test()
    print("Data {} finished".format(data_name))

    ae_result = [auc, precision, recall, f_score]
    ae_result_np = np.array(ae_result).reshape(1, -1)
    ae_result_df = pd.DataFrame(ae_result_np, columns=df_columns)
    if not os.path.exists('results'):
        os.mkdir('results')
    ae_result_df.to_csv('results/%s_%s.csv' % (data_name[:-4], 'RCA'), index=False)
