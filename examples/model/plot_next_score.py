from qlib.workflow import R
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

close_df = record.load_object('close.pkl')
label_df = record.load_object('label.pkl')
pred_df = record.load_object('pred_df.pkl')

print(close_df.head())
print(label_df.head())
print(pred_df.head())