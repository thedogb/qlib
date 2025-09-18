import pandas as pd

label_df = pd.read_csv('label.csv')
pred_df = pd.read_csv('pred.csv')

all_df = pd.merge(label_df, pred_df, how='inner', on=['datetime', 'instrument'])
all_df = all_df.sort_values(by='score', ascending=False)
all_df.to_csv('all.csv')
