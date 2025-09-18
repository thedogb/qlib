from pprint import pprint

import qlib
import pandas as pd
from qlib.utils.time import Freq
from qlib.utils import flatten_dict
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.data import D
import pickle
import mlflow
from pathlib import Path

# init qlib
client = mlflow.tracking.MlflowClient(tracking_uri='http://localhost:5000')
run_id='6184abac734e4b2a81f611473e1e8230'
local_path = client.download_artifacts(run_id, "pred.pkl")
with Path(local_path).open("rb") as f:
    pred_score = pickle.Unpickler(f).load()
    pred_score.to_csv('pred.csv')
qlib.init(provider_uri='/home/bartender/src/qlib_bin')

min_dt = pred_score.index.get_level_values("datetime").min()
max_dt = pred_score.index.get_level_values("datetime").max()



CSI300_BENCH = "SH000300"
# Benchmark is for calculating the excess return of your strategy.
# Its data format will be like **ONE normal instrument**.
# For example, you can query its data with the code below
# `D.features(["SH000300"], ["$close"], start_time='2010-01-01', end_time='2017-12-31', freq='day')`
# It is different from the argument `market`, which indicates a universe of stocks (e.g. **A SET** of stocks like csi300)
# For example, you can query all data from a stock market with the code below.
# ` D.features(D.instruments(market='csi300'), ["$close"], start_time='2010-01-01', end_time='2017-12-31', freq='day')`


FREQ = "day"
STRATEGY_CONFIG = {
    "topk": 20,
    "n_drop": 2,
    # pred_score, pd.Series
    "signal": pred_score,
}

EXECUTOR_CONFIG = {
    "time_per_step": "day",
    "generate_portfolio_metrics": True,
}

backtest_config = {
    "start_time": '2021-01-01',
    "end_time": '2024-12-31',
    "account": 1000000,
    "benchmark": CSI300_BENCH,
    "exchange_kwargs": {
        "freq": FREQ,
        "limit_threshold": ["Or($close>=($high_limit),$is_index > 0)", "Or($close<=($low_limit),$is_index > 0)"],
        "deal_price": "close",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    },
}

# strategy object
strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
# executor object
executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
# backtest
portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj, **backtest_config)
analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
# backtest info
report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)
print(report_normal.reset_index())
print(f'周期内策略收益: {sum(report_normal["return"] - report_normal["cost"])}')
print(f'周期内基线收益: {sum(report_normal["bench"])}')

# analysis
analysis = dict()
analysis["excess_return_without_cost"] = risk_analysis(
    report_normal["return"] - report_normal["bench"], freq=analysis_freq
)
analysis["excess_return_with_cost"] = risk_analysis(
    report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=analysis_freq
)

analysis_df = pd.concat(analysis)  # type: pd.DataFrame
# log metrics
analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())
# print out results
pprint(f"The following are analysis results of benchmark return({analysis_freq}).")
pprint(risk_analysis(report_normal["bench"], freq=analysis_freq))
pprint(f"The following are analysis results of strategy return({analysis_freq}).")
pprint(risk_analysis(report_normal["return"] - report_normal['cost'], freq=analysis_freq))
pprint(f"The following are analysis results of the excess return without cost({analysis_freq}).")
pprint(analysis["excess_return_without_cost"])
pprint(f"The following are analysis results of the excess return with cost({analysis_freq}).")
pprint(analysis["excess_return_with_cost"])