import pandas as pd

entities = 'tmf_entities_scores.tsv'
out_file1 = 'entities_freq_title.tsv'
out_file2 = 'entities_freq_abstract.tsv'
out_file3 = 'entities_freq_all.tsv'

def add_to_dict(type, ent, d):
    try:
        d[ent] += 1
    except KeyError:
        d[ent] = 1

df = pd.read_csv(entities, delimiter="\t", names=['id', 'src', 'entity', 'score'])

freqs_title = {}
freqs_abs = {}
freqs_all = {}

for index,row in df.iterrows():
    print(index)

    src = row['src']
    entity = row['entity']

    if src == 'title':
        add_to_dict(src, entity, freqs_title)
    else:
        add_to_dict(src, entity, freqs_abs)
    add_to_dict(src, entity, freqs_all)

for dic, file in zip([freqs_title, freqs_abs, freqs_all],[out_file1,out_file2,out_file3]):
    with open(file, "w") as f:
        for k,v in sorted(dic.items(), key=lambda kv: kv[1], reverse=True):
            f.write(str(k)+"\t"+str(v)+"\n")
