import pandas as pd
import os, pickle

top_topics = 3

for file in os.listdir("out/"):
    file_path = os.path.join("out/", file)

    n_topics = int(file.split("_")[1])
    n_features = int(file.split("_")[2].split(".")[0])

    df = pd.read_csv(file_path)
    df = df.groupby("id", as_index=False).agg({
        "topic": lambda x: list(x),
        "prob": lambda x: list(x)
    })
    print("[{}] file read".format(file_path))

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
    print("[{}] topics extracted".format(file_path))

    pickle.dump(topic_info, open("data/topic_info_{}_{}_mod.pkl".format(n_topics, n_features), "wb"))
    print("[{}] pickle saved\n".format(file_path))
