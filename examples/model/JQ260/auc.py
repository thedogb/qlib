import pandas as pd
import numpy as np
import plotly.graph_objs as go

# 读取数据
df = pd.read_csv('all.csv', parse_dates=['datetime'])
df['date'] = df['datetime'].dt.date
df['LABEL0'] = (df['LABEL0'] > 0).astype(int)
print(df.groupby('LABEL0').count())

# 向量化 AUC 计算函数（连续标签）
def continuous_label_auc_vectorized(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    # 创建成对组合
    idx_i, idx_j = np.triu_indices(len(labels), k=1)
    label_diff = labels[idx_i] - labels[idx_j]
    score_diff = scores[idx_i] - scores[idx_j]

    # 只保留 label 不相等的对
    mask = label_diff != 0
    label_diff = label_diff[mask]
    score_diff = score_diff[mask]

    total = len(label_diff)
    if total == 0:
        return None

    concordant = np.sum((label_diff > 0) & (score_diff > 0)) + \
                 np.sum((label_diff < 0) & (score_diff < 0))
    ties = np.sum(score_diff == 0)

    auc = (concordant + 0.5 * ties) / total
    return auc

# 计算每天的 AUC
daily_auc = []
for date, group in df.groupby('date'):
    auc = continuous_label_auc_vectorized(group['LABEL0'].values, group['score'].values)
    if auc is not None:
        daily_auc.append({'date': date, 'auc': auc})

auc_df = pd.DataFrame(daily_auc)

# Plotly 交互式图表
fig = go.Figure()
fig.add_trace(go.Scatter(x=auc_df['date'], y=auc_df['auc'],
                         mode='lines+markers',
                         name='Daily AUC'))

fig.update_layout(
    title='Daily Pairwise AUC (Vectorized, Continuous Labels)',
    xaxis_title='Date',
    yaxis_title='AUC',
    xaxis=dict(tickangle=45),
    template='plotly_white',
    width=1000,
    height=500
)

#fig.show()
fig.write_html("daily_auc_plot.html")

# 平均 AUC
print("平均 AUC（向量化）:", auc_df['auc'].mean())

