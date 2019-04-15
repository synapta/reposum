import pandas as pd
import os, pickle

n_top_topics = 3
n_top_similarities = 10
out_file = "out/similarities_len_topics.csv"

topic_files = []
topic_files_grouped = []
docs_per_topic = []

for file in os.listdir("out/"):
    if not file.startswith("probs"):
        continue

    file_path = os.path.join("out/", file)

    print("reading {}...".format(file_path))
    df = pd.read_csv(file_path)
    topic_files.append(df)
    df = df.drop(["prob"], axis=1)
    dd = df.groupby("id", as_index=False).agg({
        "topic": lambda x: list(x)
    })
    topic_files_grouped.append(dd)

    dd = df.groupby("topic", as_index=False).agg({
        "id": lambda x: list(x)
    })

    numbers = []
    for index,row in dd.iterrows():
        numbers.append(len(row['id']))
    docs_per_topic.append(numbers)

print("reading US file...")
df_id = pd.read_excel(
    "../data/tesi_US/US_PhD_dissertations.xlsx",
    usecols=[13,24],
    names=['id','abstract']
)
ids = set(topic_files[0]['id'])

with open(out_file, "w") as f:
    f.write("id,similar_id,similarity\n")
    for num, id_current in enumerate(ids):
        print(num)
        similarities = {}

        for index1,file in enumerate(topic_files):
            file_n = topic_files_grouped[index1]
            hot_topics = file_n[file_n['id']==id_current].iloc[0]['topic']
            hot_file = file[file['topic'].isin(hot_topics[:n_top_topics])]
            hot_file = hot_file.drop("prob", axis=1)
            #hot_file = hot_file.groupby("id", as_index=False).agg({
            #    "topic": lambda x: list(x)
            #})

            #print("iterating file {}...".format(index1))
            df_ids = list(set(hot_file['id']))
            #for index2, row in hot_file.iterrows():
            #for index2,id in enumerate(df_ids):
            for i in range(len(df_ids)):
                if str(i).endswith("3"):
                    i = i + 7
                    if i >= len(df_ids):
                        break

                #id = row['id']
                #if id == id_current:
                #    continue

                #if row['topic'] not in hot_topics:
                #    continue

                try:
                    #similarities[row['id']] += 1
                    #similarities[id] += 1/(docs_per_topic[index1][df_topics[index2]])
                    similarities[df_ids[i]] += 1/len(hot_topics)
                except KeyError:
                    #similarities[row['id']] = 1
                    #similarities[id] = 1/(docs_per_topic[index1][df_topics[index2]])
                    similarities[df_ids[i]] = 1/len(hot_topics)

        cnt = 0
        for id,similarity in sorted(similarities.items(), key=lambda kv: kv[1], reverse=True):
            f.write(str(id_current)+","+str(id)+","+str(similarity)+"\n")

            cnt+=1
            if cnt == n_top_similarities:
                break
