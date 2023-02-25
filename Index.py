from model_2 import analys
import pandas as pd

data_set = pd.read_csv("new_data.csv")
data_analysis = []
da = pd.DataFrame(index=['empty','sadness','enthusiasm','neutral','worry','surprise','love','fun','hate','happiness','boredom','relief','anger'])
for  row in data_set.iterrows():
    da[f"{row[0]}"] = analys(row[1].ENTRY.split('.'))
da.to_csv("Analysis_Data.csv")
