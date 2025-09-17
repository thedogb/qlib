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

batch_size = 2
seq_len = 60
n_features = 6
n_label = 20

x = torch.rand(batch_size, seq_len, n_features)
y_true = torch.rand(batch_size, n_label)

model = CustomTransformerRegressor(seq_len, n_features, n_label)