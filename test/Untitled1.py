#!/usr/bin/env python
# coding: utf-8
from qlib.contrib.model.pytorch_transformer import Transformer, PositionalEncoding
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class CustomTransformerRegressor(nn.Module):
    def __init__(self, seq_len, n_features, n_label, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        # 配置 Transformer
        config = BertConfig(
            hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=nhead,
            intermediate_size=d_model * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )

        # 使用 BertModel 的 encoder 部分
        self.transformer = BertModel(config)

        # 输入投影，将 n_features 映射到 d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # 输出 head
        self.output_proj = nn.Linear(d_model, n_label)

    def forward(self, x, attention_mask=None):
        """
        x: [batch_size, seq_len, n_features]
        attention_mask: [batch_size, seq_len] 可选
        """
        # 输入投影
        x = self.input_proj(x)  # [batch_size, seq_len, d_model]

        # Hugging Face Transformer 默认输入 shape: [batch_size, seq_len, hidden_size]
        outputs = self.transformer(inputs_embeds=x, attention_mask=attention_mask)

        # 使用最后一个 token 或者 CLS token 做回归
        # Hugging Face BertModel 默认返回 last_hidden_state: [batch_size, seq_len, hidden_size]
        # 这里取 [CLS] token = first token
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, d_model]

        y = self.output_proj(cls_output)  # [batch_size, n_label]
        return y

if __name__ == '__main__':
    from pprint import pprint
    from pathlib import Path
    import pandas as pd

    MARKET = "csi300"
    BENCHMARK = "SH000300"
    EXP_NAME = "tutorial_exp"

    import qlib

    qlib.init()


    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.processor import ZScoreNorm, Fillna



    dataset_config_str = f'''
    data_handler_config:
      start_time: 2008-01-01
      end_time: 2010-12-01
      fit_start_time: 2008-01-01
      fit_end_time: 2009-12-31
      instruments: {MARKET}
      infer_processors:
        - class: RobustZScoreNorm
          kwargs:
            fields_group: feature
            clip_outlier: true
        - class: Fillna
          kwargs:
            fields_group: feature
      learn_processors:
        - class: DropnaLabel
        - class: CSRankNorm
          kwargs:
            fields_group: label
      label:
        - [ "Ref($close, -1) / $close - 1", "Ref($close, -2) / $close - 1", "Ref($close, -3) / $close - 1", "Ref($close, -4) / $close - 1", "Ref($close, -5) / $close - 1", "Ref($open, -1) / $close - 1", "Ref($open, -2) / $close - 1", "Ref($open, -3) / $close - 1", "Ref($open, -4) / $close - 1", "Ref($open, -5) / $close - 1", "Ref($high, -1) / $close - 1", "Ref($high, -2) / $close - 1", "Ref($high, -3) / $close - 1", "Ref($high, -4) / $close - 1", "Ref($high, -5) / $close - 1", "Ref($low, -1) / $close - 1", "Ref($low, -2) / $close - 1", "Ref($low, -3) / $close - 1", "Ref($low, -4) / $close - 1", "Ref($low, -5) / $close - 1" ]
        - [ 'close_next1', 'close_next2', 'close_next3', 'close_next4', 'close_next5', 'open_next1', 'open_next2', 'open_next3', 'open_next4', 'open_next5', 'high_next1', 'high_next2', 'high_next3', 'high_next4', 'high_next5', 'low_next1', 'low_next2', 'low_next3', 'low_next4', 'low_next5' ]
    '''

    import yaml
    handler_kwargs = yaml.safe_load(dataset_config_str)['data_handler_config']
    handler_conf = {
        "class": "Alpha360",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": handler_kwargs,
    }
    pprint(handler_conf)



    from qlib.utils import init_instance_by_config


    dataset_conf = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_conf,
            "segments": {
                "train": ("2008-01-01", "2008-12-31"),
                "valid": ("2009-01-01", "2009-03-31"),
                "test": ("2009-04-01", "2009-12-31"),
            },
        },
    }



    dataset = init_instance_by_config(dataset_conf)
    print(dataset.prepare('train'))



    from  qlib.contrib.model.pytorch_transformer import TransformerModel




    model_config = {
            'class': 'TransformerModel',
            'module_path': 'qlib.contrib.model.pytorch_transformer',
            'kwargs': {
                'd_feat': 6,
                'd_model': 1,
                'nhead': 1,
                'num_layers': 1,
                'dropout': 0.5,
                'seed': 42,
                'batch_size': 2,
                'n_epochs': 1
            }
    }




    # model = init_instance_by_config(model_config)




    df_train, df_valid, df_test = dataset.prepare(
                ["train", "valid", "test"],
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_L,
            )




    x_train, y_train = df_train["feature"], df_train["label"]




    import numpy as np
    x_train_values = x_train.values
    y_train_values = np.squeeze(y_train.values)




    x_train_values.shape




    y_train_values.shape




    indices = np.arange(len(x_train_values))




    indices




    np.random.shuffle(indices)




    indices




    import torch
    batch_size =4
    i = 0
    feature = torch.from_numpy(x_train_values[indices[i : i + batch_size]]).float()
    label = torch.from_numpy(y_train_values[indices[i : i + batch_size]]).float()




    feature.shape



    from qlib.contrib.model.pytorch_transformer import Transformer, PositionalEncoding
    import torch
    import torch.nn as nn
    import torch.optim as optim

    model = Transformer(6, 8, 4, 2, 0)
    # pre = model.model(feature)




    model.train()

    pred = model(feature)
    print(pred)

