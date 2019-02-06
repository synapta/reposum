from elasticsearch import Elasticsearch
import dataset_utils as dsu
import pandas as pd
import re, json

index_name = 'wiki-test'
num_results = 20

es_query = {
  "query": {
    "bool": {
      "must": [{
        "simple_query_string": {
          "query": "",
          "default_operator": "or",
          "fields": ["contesti"]
        }
      }]
    }
  },
  "_source": {
    "excludes": ["contesti"]
  },
  "size": num_results
}

def parse_es_results(es_dict, split_list=True):
	query_hits = []
	for hit in es_dict['hits']['hits']:
		query_hits.append(hit['_source']['title'])
	if split_list:
		split_index = int(num_results/2)
		return query_hits[:split_index], query_hits[split_index:]
	else:
		return query_hits

####################################################################

es = Elasticsearch('localhost', port=9205)

data = dsu.read_dataset_US(nrows=None)

tmf_entities = {
	'doc_id': [],
	'title_best': [],
	'title_worst': [],
	'abstract_best': [],
	'abstract_worst': [],
}

for index,row in data.iterrows():
	print(index)
	title = row[' Titolo '].replace("\"", "'")
	doc_id = row[' ID documento ProQuest '] 
	abstract = row[' Abstract '].replace("\"", "'")

	process_title = False
	process_abstract = False

	if '***NO TITLE PROVIDED***' not in title:
		es_query['query']['bool']['must'][0]['simple_query_string']['query'] = title
		res_title = es.search(index=index_name, body=es_query)
		title_hits_best, title_hits_worst = parse_es_results(res_title)
		tmf_entities['doc_id'].append(doc_id)
		tmf_entities['title_best'].append('\n'.join(title_hits_best))
		tmf_entities['title_worst'].append('\n'.join(title_hits_worst))
		process_title = True
	else:
		continue

	if abstract != '  Nessun elemento disponibile. ' and abstract != '  Abstract Not Available. ' and abstract != '  Abstract not Available. ' and abstract != '  Abstract not available. ':
		es_query['query']['bool']['must'][0]['simple_query_string']['query'] = abstract
		res_abstract = es.search(index=index_name, body=es_query)
		abstract_hits_best, abstract_hits_worst = parse_es_results(res_abstract)
		tmf_entities['abstract_best'].append('\n'.join(abstract_hits_best))
		tmf_entities['abstract_worst'].append('\n'.join(abstract_hits_worst))
		process_abstract = True
	else:
		tmf_entities['abstract_best'].append(None)
		tmf_entities['abstract_worst'].append(None)

print("--- FINISH ---")

df = pd.DataFrame(tmf_entities)
df.to_csv("tmf_entities.csv", index=None)

			