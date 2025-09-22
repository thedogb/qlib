# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from typing import Text, Union
import copy
import math
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


# qrun examples/benchmarks/Transformer/workflow_config_transformer_Alpha360.yaml ”


class TstModel(Model):
    def __init__(
            self,
            d_feat: int = 6,
            seq_len: int = 60,
            d_price: int = 4,
            d_model: int = 64,
            batch_size: int = 2048,
            nhead: int = 2,
            num_layers: int = 2,
            dropout: float = 0,
            step_dim: int = 8,
            out_steps: int = 5,
            n_epochs=100,
            lr=0.0001,
            metric="",
            early_stop=5,
            loss="mse",
            optimizer="adam",
            reg=1e-3,
            n_jobs=10,
            GPU=0,
            seed=None,
            **kwargs,
    ):
        # set hyper-parameters.
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.n_jobs = n_jobs
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.logger = get_module_logger("TstModel")
        self.logger.info("Naive Tst:" "\nbatch_size : {}" "\ndevice : {}".format(self.batch_size, self.device))

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = PatchTST(d_feat, seq_len, n_price=d_price,
                              d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout, step_dim=step_dim,
                              out_steps=out_steps, device=self.device)

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        self.model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i: i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i: i + self.batch_size]]).float().to(self.device)

            pred = self.model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i: i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i: i + self.batch_size]]).float().to(self.device)

            with torch.no_grad():
                pred = self.model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
            self,
            dataset: DatasetH,
            evals_result=dict(),
            save_path=None,
    ):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        self.label_cols = y_train.columns

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            self.logger.info("train %.6f, valid %.6f, best_score %.6f" % (train_score, val_score, best_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size

            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)

            with torch.no_grad():
                pred = self.model(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.DataFrame(np.concatenate(preds), index=index, columns=self.label_cols)


from qlib.contrib.model.pytorch_transformer import Transformer, PositionalEncoding
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel, PatchTSTForPrediction
from transformers.models.patchtst.modeling_patchtst import PatchTSTPredictionHead


class PatchTST(nn.Module):
    def __init__(self, n_features=6, seq_len=60, distribution_output='student_t', n_price=4, d_model=64, nhead=4,
                 num_layers=2, dropout=0.1, step_dim=8, out_steps=5, device=None):
        super().__init__()

        self.n_features = n_features
        self.seq_len = seq_len
        self.distribution_output = distribution_output
        self.n_price = n_price
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.step_dim = step_dim
        self.out_steps = out_steps

        # 配置 Transformer
        config = PatchTSTConfig(
            num_input_channels=n_features,
            context_length=seq_len,
            distribution_output=distribution_output,
            patch_length=1,
            patch_stride=1,
            num_hidden_layers=num_layers,
            hidden_size=d_model,
            num_attention_heads=nhead,
            channel_attention=False,
            ffn_dim=512,
            norm_type='batchnorm',
            norm_eps=1e-05,
            attention_dropout=0.,
            positional_dropout=0.,
            ff_dropout=0.,
            bias=True,
            activation_function='gelu',
            pre_norm=True,
            positional_encoding_type='sincos',
            use_cls_token=False,
            init_std=0.02,
            share_projection=True,
            scaling='std',
            pooling_type='',
            head_dropout=dropout,
            prediction_length=out_steps,
            num_parallel_samples=100

        )

        # 使用 BertModel 的 encoder 部分
        self.transformer = PatchTSTForPrediction(config).to(device)


        # 输出 head
        self.step_embed = nn.Embedding(out_steps, step_dim)
        self.output_proj = nn.Linear(n_features, n_price)
        self.print = True

    def forward(self, x):
        """
        x: [batch_size, seq_len, n_features]
        attention_mask: [batch_size, seq_len] 可选
        """
        # 输入投影
        if self.print:
            print(x)
            print('x shape:', x.shape)
        x = x.reshape(len(x), self.n_features, -1).permute(0, 2, 1)  # [B, seq_len, d_fea]
        if self.print:
            print('x reshape: ', x.shape)
        outputs = self.transformer(
            past_values=x
        )  # [B, out_step, n_model]

        if self.print:
            print(outputs)

        x_exp = outputs.prediction_outputs # [B, out_steps, d_fea]
        batch_size = x_exp.size(0)  # x_last: [B, d_model] -> [B, 64]

        y = self.output_proj(x_exp)  # [batch_size, out_steps, n_price]
        if self.print:
            print('y shape: ', y.shape)
        y_perm = y.permute(0, 2, 1)  # [B, n_price, out_steps]
        if self.print:
            print('y_perm shape: ', y_perm.shape)

        # reshape 展开成 [B, 20]
        y_flat = y_perm.reshape(batch_size, self.n_price * self.out_steps)  # [B, 20]
        if self.print:
            print('y_flat shape: ', y_flat.shape)
        self.print = False
        return y_flat
