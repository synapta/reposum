import dataset_utils as dsu
import pandas as pd
import numpy as np
import re

phils = {"title":[],
            "creator":[],
            "university":[],
            "publisher":[],
            "year":[],
            "abstract":[],
            "type":[],
            "subject":[],
            "id":[],
            "philosophy":[]}
no_phils = {"title":[],
            "creator":[],
            "university":[],
            "publisher":[],
            "year":[],
            "abstract":[],
            "type":[],
            "subject":[],
            "id":[],
            "philosophy":[]}

def scan_philosophy(data, abs):
    for index,row in data.iterrows():
        print(index)

        if re.search(r'[P|p]hilosop',str(row['argomento'])) is not None:
            append_data(phils, row, abs)
            continue

        try:
            subj = re.sub("^\s","",row["argomento"])
        except TypeError:
            append_data(no_phils, row, abs)
            continue

        append_data(no_phils, row, abs)

def append_data(dictionary, row, abs):
    dictionary['title'].append(row['titolo'])
    dictionary['creator'].append(row['autore'])
    dictionary['university'].append(row['univ'])
    dictionary['publisher'].append(row['publisher'])
    dictionary['year'].append(row['anno'])
    if abs:
        dictionary['abstract'].append(row['abs'])
    else:
        dictionary['abstract'].append("")
    dictionary['type'].append(row['tipo'])
    dictionary['subject'].append(row['argomento'])
    dictionary['id'].append(row['id'])
    dictionary['philosophy'].append("")

scan_philosophy(dsu.read_dataset_UK_ethos(True), True)
scan_philosophy(dsu.read_dataset_UK_ethos(False), False)

philsDF = pd.DataFrame(phils)
philsDF.to_csv("data/philosophy_all.csv", index=None)
nophilsDF = pd.DataFrame(no_phils)
nophilsDF.to_csv("data/no_philosophy_all.csv", index=None)

s = np.arange(philsDF.count()[0])
np.random.shuffle(s)
philsDF = philsDF.iloc[s]

s = np.arange(nophilsDF.count()[0])
np.random.shuffle(s)
nophilsDF = nophilsDF.iloc[s]

philsDF_test = philsDF.iloc[:500]
philsDF_train = philsDF.iloc[500:]
nophilsDF_test = nophilsDF.iloc[:500]
nophilsDF_train = nophilsDF.iloc[500:philsDF_train.count()[0]*10+500]

philsDF_train.to_csv("data/philosophy_train.csv", index=None)
philsDF_test.to_csv("data/philosophy_test.csv", index=None)
nophilsDF_train.to_csv("data/nophilosophy_train.csv", index=None)
nophilsDF_test.to_csv("data/nophilosophy_test.csv", index=None)
