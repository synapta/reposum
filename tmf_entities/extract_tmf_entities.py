from elasticsearch import Elasticsearch
import dataset_utils as dsu
import re, json, pickle
import pandas as pd

index_name = 'wiki-en'
num_results_title = 10
num_results_abstract = 20
score_threshold = 30.0
out_file = "tmf_entities_scores.csv"
out_pkl = "tmf_entities_scores.pkl"

tmf_entities = {}
max_score = 0
min_score = 100

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
  "size": None
}

def update_global_scores(score, min_s, max_s):
	global max_score, min_score
	max_score = score if score > max_s else max_s
	min_score = score if score < min_s else min_s

def parse_es_results(es_dict):
	query_hits = {}
	for hit in es_dict['hits']['hits']:
		hit_score = hit['_score']
		#if hit_score > score_threshold:
			#update_global_scores(hit_score, max_score, min_score)
		query_hits[hit['_id']] = hit_score
	return query_hits

####################################################################

es = Elasticsearch('localhost', port=9205)

data = dsu.read_dataset_US(nrows=None)
print("data read")

for index,row in data.iterrows():
	print(index)
	title = row[' Titolo '].replace("\"", "'")
	doc_id = row[' ID documento ProQuest '] 
	abstract = row[' Abstract '].replace("\"", "'")

	print(title)

	if '***NO TITLE PROVIDED***' not in title:
		tmf_entities[doc_id] = {
			"title":{},
			"abstract":{}
		}

		es_query['query']['bool']['must'][0]['simple_query_string']['query'] = title
		es_query['size'] = num_results_title
		res_title = es.search(index=index_name, body=es_query)		
		tmf_entities[doc_id]['title'] = parse_es_results(res_title)
	else:
		continue

	if abstract != '  Nessun elemento disponibile. ' and abstract != '  Abstract Not Available. ' and abstract != '  Abstract not Available. ' and abstract != '  Abstract not available. ':
		es_query['query']['bool']['must'][0]['simple_query_string']['query'] = abstract
		es_query['size'] = num_results_abstract
		res_abstract = es.search(index=index_name, body=es_query)
		tmf_entities[doc_id]['abstract'] = parse_es_results(res_abstract)

print("--- FINISH ---")

with open(out_file, "w") as f:
	for doc_id, inner_dict in tmf_entities.items():
		for entity, score in inner_dict['title'].items():
			#norm_score = (score - min_score) / (max_score - min_score)
			#tmf_entities[doc_id]['title'][entity] = norm_score
			f.write("{},{},{},{}\n".format(doc_id, "title", entity, score))

		for entity, score in inner_dict['abstract'].items():
			#norm_score = (score - min_score) / (max_score - min_score)
			#tmf_entities[doc_id]['abstract'][entity] = norm_score
			f.write("{},{},{},{}\n".format(doc_id, "abstract", entity, score))

#with open(out_pkl, "wb") as f:
#	pickle.dump(tmf_entities, f)