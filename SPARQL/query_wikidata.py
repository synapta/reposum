import sparql_utils as spqlu
import requests, pprint
import pandas as pd

url = 'https://query.wikidata.org/sparql'

P_names = {}
Q_names = {}
explored = set()

#stack = 0

def parse_result(page, prev_page_name, query_res, type):
    P = [t[0] for t in spqlu.useful_props[type]]
    P_recurs = [t[2] for t in spqlu.useful_props[type]]
    #d = {}
    triples = []
    for row in query_res:
        prop_id = row['wd']['value'].split("/")[-1]
        if prop_id in P:
            recurs = P_recurs[P.index(prop_id)]
            if recurs is not None:
                #global stack
                #stack += 1
                #print(" "*stack,"[{}] found a recursive element: {}".format(stack,row['ooLabel']['value']))
                recur_res = make_request(row['o']['value'].split("/")[-1], '_'.join(row['ooLabel']['value'].split()), recurs)
                #print("recur:",recur_res)
                if recur_res == "EXP" or recur_res == []:
                    triples.append((prev_page_name, '_'.join(row['wdLabel']['value'].lower().split()), '_'.join(row['ooLabel']['value'].split())))
                else:
                    triples.append((prev_page_name, '_'.join(row['wdLabel']['value'].lower().split()), '_'.join(row['ooLabel']['value'].split())))
                    triples.extend(recur_res)
                #print(triples)
                #print("\n")
                #stack = stack -1
            else:
                triples.append((prev_page_name, '_'.join(row['wdLabel']['value'].lower().split()), '_'.join(row['ooLabel']['value'].split())))
    #return d
    return triples

def make_request(page_id, prev_page_name, type):
    #global stack
    if page_id not in explored or type == "philosopher":
        #print(" "*stack, " querying {}[{}]...".format(page_id,type))
        r = requests.get(
            url,
            params = {
                'format': 'json',
                'query': spqlu.wikidata_query%(page_id)
            }
        )
        #pprint.pprint(r.json())
        #input()
        explored.add(page_id)
        return parse_result(page_id, prev_page_name, r.json()['results']['bindings'], type)
    else:
        #print(" * already explored: {}[{}]".format(page_id,type))
        return "EXP"

#def explore_philosopher(phil_Q):
#    return make_request(phil_Q, "philosopher")

wikidata_props = {}
errors = set()
df = pd.read_csv("philosophers.csv")
for index, row in df.iterrows():

    wikiQ = row['item']
    Qname = row['itemLabel']

    wikiQ = wikiQ.split("/")[-1]

    print("[{} - {}] ({})".format(wikiQ, Qname, index))

    #phil_triples = make_request("Q9465", "philosopher")
    #phil_triples = make_request("Q185", "mbare", "philosopher")
    phil_triples = make_request(wikiQ, '_'.join(Qname.split()), "philosopher")
    if phil_triples != "EXP":
        with open ("wikidata_triples_new.csv", "a") as f:
            for triple in phil_triples:
                try:
                    a,b,c = triple
                    f.write(str(a)+"\t"+str(b)+"\t"+str(c)+"\n")
                except ValueError:
                    print("ValueError at triple {}".format(triple))
                    errors.add(wikiQ)

    print("saved {} triples".format(len(phil_triples)))

with open("errors.csv", "w") as f:
    for err in errors:
        f.write(err+"\n")
