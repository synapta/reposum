import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

entities_df = pd.read_csv("data/tmf_entities.csv")
entities_df.fillna(value="", inplace=True)

G = nx.Graph()

for index,row in entities_df.iterrows():
	print(index)
	doc_id = row['doc_id']
	G.add_node(doc_id, type='thesis')

	title_best_ents = row['title_best'].split('\n')
	title_worst_ents = row['title_worst'].split('\n')

	for ent in title_best_ents:
		G.add_node(ent, type='entity')
		G.add_edge(doc_id,ent, weight=2)
	for ent in title_worst_ents:
		G.add_node(ent, type='entity')
		G.add_edge(doc_id,ent, weight=1)
	
	if len(row['abstract_best']) != 0:
		abs_best_ents = row['abstract_best'].split('\n')
		abs_worst_ents = row['abstract_worst'].split('\n')

		for ent in abs_best_ents:
			G.add_node(ent, type='entity')
			G.add_edge(doc_id,ent, weight=2)
		for ent in abs_worst_ents:
			G.add_node(ent, type='entity')
			G.add_edge(doc_id,ent, weight=1)

nx.write_gpickle(G,'graph.pkl')