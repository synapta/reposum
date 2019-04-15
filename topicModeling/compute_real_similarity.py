import networkx as nx
import os, pickle

# Could not run it because of RAM limit :(

G = nx.MultiGraph()

for file in os.listdir("data/"):
    if not file.startswith("topic"):
        continue
    if not file.endswith("mod.pkl"):
        continue

    file_path = os.path.join("data/", file)

    n_topics = int(file.split("_")[2])
    n_features = int(file.split("_")[3].split(".")[0])

    t = pickle.load(open(file_path, "rb"))
    print("[{}] topics loaded".format(file_path))

    count = 0
    for topic, levels in t.items():
        ids = levels['high']
        for id in ids:
            G.add_node(id)
        for id_current in ids:
            for id_other in ids:
                if id_current != id_other:
                    if G.has_edge(id_current, id_other):
                        w = G[id_current][id_other][0]['weight'] + 1
                        G.remove_edge(id_current, id_other)
                        G.add_edge(id_current, id_other, weight=w)
                    else:
                        G.add_edge(id_current, id_other, weight=1)
            print("[{}] adding edges for topic {} ({} documents) {}...".format(file_path,topic,len(ids),count), end="\r")
            count += 1
    print("")

nx.write_gpickle(G, "models/topics_graph.pkl")
print("graph saved")
