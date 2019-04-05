import networkx as nx
import pandas as pd
import pickle

E = pd.read_csv("Wiki_NASARI_embed_english.txt", delimiter=" ", usecols=[0], names=['word'])
print("E")
G = pickle.load(open("../SPARQL/graph_UK.pkl", "rb"))
print("G")

df = E[['word']].applymap(lambda x: str(x).lower())
pages = set(df['word'])

npreslist = []

pres = 0
npres = 0
for n in G.nodes:
    n = str(n).lower()
    """if ' ' in n:
        input("spaziooooooooooo")
        n = '_'.join([ns for ns in n.split()])
        if n in pages:
            pres += 1
        else:
            npres += 1
    else:"""
    if n in pages:
        pres += 1
    else:
        npres += 1
        npreslist.append(n)
        """ps = n.split('_')
        cnt = 0
        while True:
            nps = []
            for i in range(len(ps)):
                nps.append(ps[i].capitalize())
                if i==cnt:
                    nps.extend(ps[(i+1):])
                    cnt+=1
                    break
            name = '_'.join(nps)

            if name in pages:
                pres += 1
                break
            else:
                if cnt==len(ps):
                    npres += 1
                    break"""

print("pres: {}\nnpres: {}".format(pres,npres))

with open("npres.txt", "w") as f:
    for n in npreslist:
        f.write(str(n)+"\n")
