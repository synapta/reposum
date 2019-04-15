import pandas as pd
import numpy as np
import os, pickle

df_id = pd.read_excel(
    "../data/tesi_US/US_PhD_dissertations.xlsx",
    usecols=[13,24],
    names=['id','abstract']
)

t = pd.read_csv("out/similarities_len_topics.csv")
t = t.drop(['similarity'], axis=1)
t = t.groupby("id", as_index=False).agg({
    "similar_id": lambda x: list(x)
})

s = np.arange(t.count()[0])
np.random.shuffle(s)

for index in s:
    print(df_id[df_id['id']==t.iloc[index]['id']].iloc[0]['abstract'])
    print("\n-----------------------------------------")
    print(df_id[df_id['id']==t.iloc[index]['similar_id'][0]].iloc[0]['abstract'])
    print("\n-----------------------------------------")
    print(df_id[df_id['id']==t.iloc[index]['similar_id'][1]].iloc[0]['abstract'])
    print("\n-----------------------------------------")
    print(df_id[df_id['id']==t.iloc[index]['similar_id'][2]].iloc[0]['abstract'])
    input("\n=========================================\n\n")
