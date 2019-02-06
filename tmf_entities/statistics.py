import pandas as pd

def read_entities_file():
	entities_df = pd.read_csv("data/tmf_entities.csv")
	entities_df.fillna(value="", inplace=True)
	return entities_df

def add_to_dict(data_dict, key, pos):
	try:
		data_dict[key][pos] += 1
	except KeyError:
		data_dict[key] = [1,0]

def compute_entities_frequency():
	entities_df = read_entities_file()
	entities = {}

	for index,row in entities_df.iterrows():
		print(index)

		total_entities = set()

		title_ents = row['title_best'].split('\n')
		title_ents.extend(row['title_worst'].split('\n'))

		for ent in title_ents:
			add_to_dict(entities, ent, 0)
			total_entities.add(ent)
		
		if len(row['abstract_best']) != 0:
			abs_ents = row['abstract_best'].split('\n')
			abs_ents.extend(row['abstract_worst'].split('\n'))

			for ent in abs_ents:
				add_to_dict(entities, ent, 0)
				total_entities.add(ent)
		
		for ent in total_entities:
			add_to_dict(entities, ent, 1)

	with open("data/entities_freqs.csv", "w") as out_file:
		for k,v in sorted(entities.items(), key=lambda kv: kv[1], reverse=True):
			out_file.write(k+","+str(v[0])+","+str(v[1])+"\n")


if __name__ == '__main__':
	compute_entities_frequency()

	
