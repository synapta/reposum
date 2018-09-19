import dataset_utils as dsu
import pandas as pd
import re

phils = {"title":[],
            "creator":[],
            "university":[],
            "publisher":[],
            "year":[],
            "abstract":[],
            "type":[],
            "subject":[],
            "id":[]}
no_phils = {"title":[],
            "creator":[],
            "university":[],
            "publisher":[],
            "year":[],
            "abstract":[],
            "type":[],
            "subject":[],
            "id":[]}

def scan_philosophy(data):
    for index,row in data.iterrows():
        print(index)

        if re.search(r'[P|p]hilosop',str(row['argomento'])) is not None:
            append_data(phils, row)
            continue

        try:
            subj = re.sub("^\s","",row["argomento"])
        except TypeError:
            append_data(no_phils, row)
            continue

        if re.match(r"^[0-9\.]+$", subj) is not None:
            if subj[0] != "1":
                append_data(no_phils, row)
            else:
                append_data(phils, row)
        else:
            append_data(no_phils, row)

def append_data(dictionary, row):
    dictionary['title'].append(row['titolo'])
    dictionary['creator'].append(row['autore'])
    dictionary['university'].append(row['univ'])
    dictionary['publisher'].append(row['publisher'])
    dictionary['year'].append(row['anno'])
    dictionary['abstract'].append(row['abs'])
    dictionary['type'].append(row['tipo'])
    dictionary['subject'].append(row['argomento'])
    dictionary['id'].append(row['id'])

scan_philosophy(dsu.read_dataset_UK_id(True))
scan_philosophy(dsu.read_dataset_UK_id(False))

pd.DataFrame(phils).to_csv("data/philosophy.csv", index=None)
pd.DataFrame(no_phils).to_csv("data/no_philosophy.csv", index=None)
