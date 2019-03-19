import matplotlib.pyplot as plt
import networkx as nx

G = nx.read_gpickle('data/graph.pkl')

nx.draw_random(G, with_labels=False, font_weight='bold')
plt.show()