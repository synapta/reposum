import pandas as pd
import os, pickle

lda_file = ""
top_topics = 5

n_topics = lda_file.split("_")[1]
n_features = lda_file.split("_")[2].split(".")[0]

df = pd.read_csv(lda_file)
df = df.groupby("id", as_index=False).agg({
    "topic": lambda x: list(x),
    "prob": lambda x: list(x)
})

topic_info = {}
for i in range(n_topics):
    topic_info[i] = {
        "high": [],
        "med": [],
        "low": []
    }

for index, row in df.iterrows():
    id = row['id']
    topics = row['topic'][0:top_topics]
    probs = row['prob'][0:top_topics]

    for topic, prob, level in zip(topics, probs, ['high', 'med', 'low']):
        topic_info[topic][level].append(id)

pickle.dump(topic_info, open("data/topic_info_{}_{}.pkl".format(n_topics, n_features), "wb"))
