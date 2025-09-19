from qlib.workflow import R
from qlib.data import D
import qlib
from qlib.workflow.recorder import Recorder
from qlib.workflow.task.utils import list_recorders

qlib.init(provider_uri='/home/bartender/src/qlib_bin', region='cn', exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": "http://localhost:5000",
                "default_exp_name": "jq260_3d_classify"
            }
        })

rid = 'e009011d9abf4c1294ea104528976622'
record = R.get_recorder(recorder_id=rid)

pred_df = record.load_object('pred_df.pkl')
print(pred_df.head())

df = D.features(
    ["SH601216"],
    ["$open", "$high", "$low", "$close", "$factor"],
    start_time="2020-05-01",
    end_time="2020-05-31",
)

import plotly.graph_objects as go
import plotly.io as pio

# pio.renderers.default = "notebook"
fig = go.Figure(
    data=[
        go.Candlestick(
            x=df.index.get_level_values("datetime"),
            open=df["$open"],
            high=df["$high"],
            low=df["$low"],
            close=df["$close"],
        )
    ]
)
fig.show()