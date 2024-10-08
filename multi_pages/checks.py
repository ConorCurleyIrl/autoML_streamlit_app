

import pandas as pd
df = pd.read_csv("data/mushroom_dataset.csv", index_col=None)
df.to_parquet('data/mushrooms.parquet')





#df = pd.read_parquet("data/iris.csv", index_col=None)