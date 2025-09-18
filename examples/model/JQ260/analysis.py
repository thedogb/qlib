from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
import qlib
from qlib.config import REG_CN
import sys

# 初始化 QLib，默认使用中国A股市场数据
qlib.init(provider_uri='/home/bartender/src/qlib_bin')

rid = 'b5152152f95545dabf03e5c1f5a188d7'
EXP_NAME = '446354448583429913'
uri_folder = "mlruns"


# load recorder
recorder = R.get_recorder(recorder_id=rid, experiment_id=EXP_NAME)


# load previous results
pred_df = recorder.load_object("pred.pkl")
label_df = recorder.load_object("label.pkl")
print(pred_df.head())
pred_df.to_csv('pred.csv')
label_df.to_csv('label.csv')
report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

# Previous Model can be loaded. but it is not used.
loaded_model = recorder.load_object("params.pkl")
print(type(loaded_model))
