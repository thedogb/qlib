# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

from linecache import cache

import numpy as np
import pandas as pd
from typing import Text, Union
import copy
import math

from mpmath import zeros

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
            out_steps: int = 5,
            loss_alpha: float= 1,
            loss_beta: float= 1,
            batch_size: int = 2048,
            n_epochs=100,
            lr=0.0001,
            metric="",
            early_stop=5,
            optimizer="adam",
            reg=1e-3,
            n_jobs=10,
            GPU=0,
            seed=None,
            **kwargs,
    ):
        # set hyper-parameters.
        self.d_feat = d_feat
        self.out_steps = out_steps
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.n_jobs = n_jobs
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.logger = get_module_logger("QingZiTstModel")
        self.logger.info("Naive Tst:" "\nbatch_size : {}" "\ndevice : {}".format(self.batch_size, self.device))



        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)


        self.model = myPromptTs(n_feature=d_feat, n_pred=out_steps,device=self.device)

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
        # mask = ~torch.isnan(label)

        return  myPromptTs.interval_loss(pred, label, self.out_steps, self.loss_alpha, self.loss_beta)

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
            # loss = self.model.interval_loss(pred, label)
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

        def gen_out_cols(label_cols):
            value_names = ['mean', 'upper', 'lower']
            columns = []

            for label in label_cols:
                for v in value_names:
                    columns.append(f'{label}_{v}')
            return columns

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
                pred = pred.reshape(pred.shape[0], -1)
            preds.append(pred)

        out_cols = gen_out_cols(self.label_cols)

        return pd.DataFrame(np.concatenate(preds), index=index, columns=out_cols)


from qlib.contrib.model.pytorch_transformer import Transformer, PositionalEncoding
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel, PatchTSTForPrediction
from transformers.models.patchtst.modeling_patchtst import PatchTSTPredictionHead

import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    """
    通用位置编码模块
    - mode='sine'	 正弦位置编码（1D/2D，动态生成）
    - mode='learned' 可学习位置编码（1D/2D，固定大小）
    - is_2d=True  时，forward 输入固定为 [B, H, W, C]
    - is_2d=False 时，forward 输入固定为 [B, L, D]
    """

    def __init__(self, d_model, max_len=100, mode='sine', is_2d=False):
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        self.is_2d = is_2d
        self.max_len = max_len

        if mode == 'learned':
            if is_2d:
                # 存储为 [1, H, W, C]
                self.pe = nn.Parameter(torch.zeros(1, max_len, max_len, d_model))
                nn.init.trunc_normal_(self.pe, std=0.02)
            else:
                # 存储为 [1, L, D]
                self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
                nn.init.trunc_normal_(self.pe, std=0.02)

    def _build_1d_sine_pos_embed(self, length, device):
        pe = torch.zeros(length, self.d_model, device=device)
        position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device).float()
            * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, L, D]

    def _build_2d_sine_pos_embed(self, H, W, device):
        d_model_half = self.d_model // 2
        pe_h = self._build_1d_sine_pos_embed(H, device).squeeze(0)[:, :d_model_half]  # [H, C/2]
        pe_w = self._build_1d_sine_pos_embed(W, device).squeeze(0)[:, :d_model_half]  # [W, C/2]

        pe = torch.zeros(1, H, W, self.d_model, device=device)  # [1, H, W, C]
        pe[:, :, :, :d_model_half] = pe_h.unsqueeze(1).repeat(1, W, 1)
        pe[:, :, :, d_model_half:] = pe_w.unsqueeze(0).repeat(H, 1, 1)
        return pe

    def forward(self, x):
        """
        is_2d=True	-> 输入 [B, H, W, C]
        is_2d=False -> 输入 [B, L, D]
        """
        if self.is_2d:
            B, H, W, C = x.shape
            assert C == self.d_model, f"输入最后一维 {C} 必须等于 d_model {self.d_model}"
            if self.mode == 'sine':
                pe = self._build_2d_sine_pos_embed(H, W, x.device)
            else:  # learned
                pe = self.pe[:, :H, :W, :]
        else:
            B, L, D = x.shape
            assert D == self.d_model, f"输入最后一维 {D} 必须等于 d_model {self.d_model}"
            if self.mode == 'sine':
                pe = self._build_1d_sine_pos_embed(L, x.device)
            else:  # learned
                pe = self.pe[:, :L, :]

        return x + pe


class FlexibleTransformerEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 d_model,
                 output_dim,
                 num_layers=6,
                 nhead=8,
                 dim_feedforward=2048,
                 dropout=0.0,
                 max_len=500,
                 batch_first=True,
                 pos_mode='sine',
                 is_2d=False):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        self.is_2d = is_2d
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.pos_encoder = PositionalEmbedding(d_model, max_len, mode=pos_mode, is_2d=is_2d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, output_dim) if output_dim != d_model else nn.Identity()

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        if self.is_2d and x.dim() == 4:
            # 展平成 [B, HW, D]
            B, H, W, D = x.shape
            x = x.view(B, H * W, D)

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.output_proj(x)


class FlexibleTransformerDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 d_model,
                 output_dim,
                 num_layers=6,
                 nhead=8,
                 dim_feedforward=2048,
                 dropout=0.0,
                 max_len=500,
                 batch_first=True,
                 pos_mode='sine',
                 is_2d=False):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        self.is_2d = is_2d
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.pos_encoder = PositionalEmbedding(d_model, max_len, mode=pos_mode, is_2d=is_2d)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model,
                                     output_dim)  # if output_dim != d_model else nn.Identity() Linear need to scale res

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt = self.input_proj(tgt)
        tgt = self.pos_encoder(tgt)
        out = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.output_proj(out)


import torch
import torch.nn as nn
import math
import yaml
import torch
import torch.nn.functional as F


class myPromptTs(nn.Module):
    def __init__(self, n_feature=6, n_pred=5, device=torch.device('cpu')):
        """
        初始化myOCHL模型

        Args:
            config_path: YAML配置文件路径
            **kwargs: 其他参数，会覆盖配置文件中的参数
        """
        super().__init__()

        self.n_feature = n_feature
        self.n_pred = n_pred
        self.device = device
        self.embedding = nn.Embedding(num_embeddings=100, embedding_dim=512).to(device)
        # self.prompt_decoder = FlexibleTransformerDecoder(
        #     input_dim=512, d_model=512, output_dim=512, num_layers=2, nhead=8,
        #     dim_feedforward=2048, dropout=0.1, max_len=50, batch_first=True, pos_mode='sine', is_2d=False,
        # ).to(device)

        self.encoder = FlexibleTransformerEncoder(
            input_dim=n_feature, d_model=512, output_dim=512, num_layers=4, nhead=8,
            dim_feedforward=2048, dropout=0.1, max_len=500, batch_first=True, pos_mode='sine', is_2d=False,
        ).to(device)
        self.decoder = FlexibleTransformerDecoder(
            input_dim=512, d_model=512, output_dim=12, num_layers=4, nhead=8, dim_feedforward=2048,
            dropout=0.1, max_len=500, batch_first=True, pos_mode='sine', is_2d=False,
        ).to(device)

        self.relu = nn.ReLU().to(device)
    def preprocess(self, x0):
        scale, x = x0[:, -1, 1].clone(), x0.clone() # scale 最新一天的close
        x[:, :, :5] = x[:, :, :5] / scale[:, None, None] - 1 # 前 5维 除以close -1， 推测前5维是ochl vwap,
        x[:, :, 5:7] = torch.clip(torch.log((1 + x[:, :, 5:7]) / x[:, -1, 5:7][:, None, :]), -3.0, 3.0) # 这两维估计是成交量和成交金额
        return x, scale

    def postprocess(self, output):
        # output：shape = (batch, N_pred, 4, 3)
        #  high = max(open, close) + 0.1 * high
        #  low = min(open, close) - 0.1 * low
        temp_2_0 = torch.maximum(output[:, :, 0, 0], output[:, :, 1, 0]) + 0.1 * self.relu(output[:, :, 2, 0])
        temp_3_0 = torch.minimum(output[:, :, 0, 0], output[:, :, 1, 0]) - 0.1 * self.relu(output[:, :, 3, 0])

        # [1] = price + 0.1 * relu([1]) 所有值的上界
        # [2] = price - 0.1 * relu([2]) 所有值的下界
        temp_all_1 = output[:, :, :, 0] + 0.1 * self.relu(output[:, :, :, 1])
        temp_all_2 = output[:, :, :, 0] - 0.1 * self.relu(output[:, :, :, 2])

        # 最后一次性赋值
        output = output.clone()  # 避免修改原计算图
        output[:, :, 2, 0] = temp_2_0  ## h
        output[:, :, 3, 0] = temp_3_0  ## L
        output[:, :, :, 1] = temp_all_1
        output[:, :, :, 2] = temp_all_2
        # output = (output + 1) * scale[:, None, None, None]
        return output

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (N_batch, N_feat, input_dim)
            x_prompt: prompt张量，形状为 (N_batch, N_prompt)

        Returns:
            输出张量，形状为 (N_batch, N_pred, output_dim)
        """

        # 对prompt进行embedding
        # x_prompt_encode = self.embedding(x_prompt)  # (N_batch, N_prompt, d_model)
        x = x.reshape(len(x), self.n_feature, -1).permute(0, 2, 1)  # [B, seq_len, d_fea]
        d_model = self.embedding.embedding_dim
        zeros_pos = torch.zeros(x.shape[0], self.n_pred, d_model).to(self.device)
        # x_prompt_decode = self.prompt_decoder(zeros_pos, x_prompt_encode)

        # 使用编码器处理输入序列
        # 编码器期望的输入形状是 (N_batch, N_feat, d_model)
        # x, scale = self.preprocess(x)

        memory = self.encoder(x)  # (N_batch, N_feat, d_model)
        output = self.decoder(tgt=zeros_pos, memory=memory)  # (N_batch, N_pred, output_dim)
        output = output.view(output.shape[0], output.shape[1], 4, 3)
        output = self.postprocess(output)
        return output

    @staticmethod
    def interval_loss( pred, label, n_pred = 5, alpha=1.0, beta=1.0):
        """
        pred: [batch, step, n_price, 3]  --> [center, upper, lower]
        label: [batch, step, n_price]
        alpha: 权重, 中心回归
        beta: 权重, 区间覆盖
        """
        label = label.reshape(len(label), n_pred, -1)
        center = pred[..., 0]  # 中心预测
        upper = pred[..., 1]  # 上界
        lower = pred[..., 2]  # 下界

        # 1. 中心回归损失 (MSE)
        loss_center = F.mse_loss(center, label)

        # 2. 区间覆盖损失 (label 超出区间才惩罚)
        loss_interval = torch.mean(F.relu(lower - label) + F.relu(label - upper))

        # 3. 可选：保证上下界合理（下界 <= 中心 <= 上界）
        # loss_order = torch.mean(F.relu(lower - center) + F.relu(center - upper))

        # 4. 区间不要过宽
        loss_width = torch.mean(upper - lower)

        # 总 loss
        loss = alpha * loss_center + beta * loss_interval + 0.1 * loss_width
        return loss
