from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
import qlib
from qlib.config import REG_CN
import sys
from factor_loader.JQ260_TEST import JQ260DL

import pandas as pd

import lightgbm as lgb
from graphviz import Digraph
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ndcg_score
import pandas as pd

# 假设你的 DataFrame 叫 df
def compute_metrics_per_day(df):
    results = []
    df['LABEL0'] = df['LABEL0'].fillna(0)

    for dt, group in df.groupby('datetime'):
        y_true = group['LABEL0'].values
        y_score = group['score'].values
        print(dt)
        print(group[group['LABEL0'].isna()]
)
        # AUC：需要至少有两个类别，否则会报错
        if len(set(y_true)) < 2:
            auc = None
            print(dt, 'not 2 label')
        else:
            auc = roc_auc_score(y_true, y_score)

        # NDCG@5：需要输入成 shape=(1, n_samples)
        # 按 score 排序，取 top 5 的真实标签作为 relevance
        top_k = 5
        if len(group) >= 1:
            ndcg_5 = ndcg_score([y_true], [y_score], k=5)
            ndcg_10 = ndcg_score([y_true], [y_score], k=10)
            ndcg_20 = ndcg_score([y_true], [y_score], k=20)
            ndcg_50 = ndcg_score([y_true], [y_score], k=50)
        else:
            print(dt, 'group <= 1')
            ndcg = None

        results.append({
            'datetime': dt,
            'auc': auc,
            'ndcg@5': ndcg_5,
            'ndcg@10': ndcg_10,
            'ndcg@20': ndcg_20,
            'ndcg@50': ndcg_50,
        })

    return pd.DataFrame(results)


def render_tree(booster, tree_index=0, out_file="tree0", feature_names=None):
    tree = booster.dump_model()['tree_info'][tree_index]['tree_structure']
    if feature_names is None:
        feature_names = booster.feature_name()

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def add_nodes(dot, node, node_id):
        if 'split_index' in node:
            fid = node['split_feature']
            fname = feature_names[fid] if fid < len(feature_names) else f"f{fid}"
            threshold = node['threshold']
            name = f"split{node['split_index']}\n{fname} ≤ {threshold}"
            dot.node(str(node_id), name, shape="box", style="filled", color="lightblue")
            add_nodes(dot, node['left_child'], f"{node_id}l")
            add_nodes(dot, node['right_child'], f"{node_id}r")
            dot.edge(str(node_id), f"{node_id}l", label="yes")
            dot.edge(str(node_id), f"{node_id}r", label="no")
        else:
            raw_score = node['leaf_value']
            prob = sigmoid(raw_score)
            label = f"leaf\nscore: {raw_score:.4f}\nprob: {prob:.4f}"
            dot.node(str(node_id), label, shape="ellipse", style="filled", color="lightgreen")

    dot = Digraph(format="svg")
    add_nodes(dot, tree, "0")
    output_path = dot.render(filename=out_file, format="svg", cleanup=True)
    print(f"Saved to: {output_path}")



def write_position_dict_to_file(data_dict, output_file='position_records.csv'):
    records = []

    for date, data in data_dict.items():
        position = data.position
        init_cash = data.init_cash
        cash = data.get_cash(False)
        now_account_value = position.get('now_account_value')

        for code, details in position.items():
            if code in ['cash', 'now_account_value']:
                continue  # 跳过非股票字段
            records.append({
                'date': pd.to_datetime(date),
                'code': code,
                'amount': details.get('amount'),
                'price': details.get('price'),
                'weight': details.get('weight'),
                'count_day': details.get('count_day'),
                'cash': cash,
                'now_account_value': now_account_value,
                'init_cash': init_cash,
                '_settle_type': data._settle_type
            })

    df = pd.DataFrame(records)
    df.sort_values(['date', 'code'], inplace=True)
    df.to_csv(output_file, index=False)
    print(f"Written {len(df)} records to {output_file}")


# 初始化 QLib，默认使用中国A股市场数据
qlib.init(provider_uri='/home/bartender/src/qlib_bin')

rid = 'cac81d41a35948498c9f2f3ffec1e847'
EXP_NAME = '446354448583429913'
uri_folder = "mlruns"


# load recorder
recorder = R.get_recorder(recorder_id=rid, experiment_id=EXP_NAME)


# load previous results
pred_df = recorder.load_object("pred.pkl")
label_df = recorder.load_object("label.pkl")
result = pd.merge(pred_df, label_df, on=['datetime', 'instrument'], how='inner')
print(result.head())
result=result.reset_index()
result = result[~result['instrument'].str.startswith('BJ')]
metrics = compute_metrics_per_day(result)
print(pred_df.head())
pred_df.to_csv('pred.csv')
label_df.to_csv('label.csv')
result['decile'] = pd.qcut(result['score'], q=10, labels=False) + 1
result.to_csv('pred_with_label.csv')
metrics.to_csv('metrics.csv')
report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
write_position_dict_to_file(positions, 'positions.csv')

# Previous Model can be loaded. but it is not used.
loaded_model = recorder.load_object("params.pkl")
# print(loaded_model.feature_name_)
feature_importance = loaded_model.get_feature_importance()
conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
            "JQ": {}
        }
fields, fea_name = JQ260DL.get_feature_config(conf)
feature_importance = {i: v for i,v in feature_importance.items()}
for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True):
    if v > 0:
        print(f"{k}: {v}")

import lightgbm as lgb
import lightgbm as lgb
import matplotlib.pyplot as plt

# 假设你已有 booster 对象
booster = loaded_model.model

# 可视化第0棵树
# lgb.plot_tree(booster, tree_index=0, figsize=(20, 10), show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
# plt.savefig('tree.png')
for i in range(booster.num_trees()):
    render_tree(booster, tree_index=i, out_file=f"tree/tree{i}", feature_names=fea_name)

# model_dict = loaded_model.booster_.dump_model()
# print(model_dict)
print(type(loaded_model))

