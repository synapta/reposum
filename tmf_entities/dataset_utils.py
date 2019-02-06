import pandas as pd

#row[2]: titolo
#row[5]: lingua
#row[9]: autore
#row[11]: CIA
#row[13]: ID documento
#row[14]: ISBN
#row[17]: fonte
#row[20]: keywords
#row[21]: pagine
#row[22]: dipartimento
#row[23]: data
#row[24]: abstract
#row[25]: url
#row[26]: #dissertazione/tesi
#row[27]: luogo
#row[28]: istituzione
#row[29]: num_pagine
#row[30]: subject
#row[31]: tipo di fonte
#row[32]: relatore
tesi_US = "../data/tesi_US/US_PhD_dissertations.xlsx"

#dc:title
#dc:creator
#uketdterms:institution
#dc:publisher
#dcterms:issued
#dcterms:abstract
#uketdterms:qualificationname
#dc:subject
tesi_UK_abs = "../data/tesi_UK/tab_separated_value/171215/full_data/171215_1137_Synapta_EThOS_with_abstract.tsv"
tesi_UK_abs_mod = "../data/tesi_UK/tab_separated_value/171215/full_data/171215_1137_Synapta_EThOS_with_abstract_modified.csv"
tesi_UK_nabs = "../data/tesi_UK/tab_separated_value/171215/full_data/171215_1137_Synapta_EThOS_NO_abstract.tsv"
tesi_UK_abs_id = "../data/tesi_UK/tab_separated_value/171215/full_data/ABS_id.csv"
tesi_UK_nabs_id = "../data/tesi_UK/tab_separated_value/171215/full_data/NO_ABS_id.csv"

#uketdterms:ethosid <----NB
#dc:title <---- NB
#dc:creator
#uketdterms:institution
#dc:publisher
#dcterms:issued
#dcterms:abstract <---- NB
#dc:type
#uketdterms:qualificationname
#uketdterms:qualificationlevel
#dc:identifier
#dc:source
#dc:subjectxsi
#dc:subject <---- NB
tesi_UK_abs_ethos = "../data/tesi_UK/tab_separated_value/Synapta_EThOS.tsv"
tesi_UK_nabs_ethos = "../data/tesi_UK/tab_separated_value/Synapta_EThOS_No_Abstract.tsv"

def read_dataset_US(**kwargs):
    return pd.read_excel(tesi_US, **kwargs)

def read_dataset_UK(abs=True):
    if abs:
        return pd.read_csv(tesi_UK_abs_mod)
    else:
        return pd.read_csv(tesi_UK_nabs,
                            sep="\t",
                            index_col=False,
                            header=0,
                            names=["titolo","autore","univ","publisher","anno","abs","tipo","argomento"])

def read_dataset_UK_id(abs=True):
    if abs:
        return pd.read_csv(tesi_UK_abs_id)
    else:
        return pd.read_csv(tesi_UK_nabs_id)

def read_dataset_UK_ethos(abs=True):
    if abs:
        return pd.read_csv(tesi_UK_abs_ethos, delimiter="\t", skiprows=1, names=["id", "titolo","autore","univ","publisher","anno","abs","tipo","qname", "qlevel", "identifier", "source", "sxsi", "argomento", "unnamed"])
    else:
        return pd.read_csv(tesi_UK_nabs_ethos, delimiter="\t", skiprows=1, names=["id","titolo","autore","univ","publisher","anno","tipo","qname", "qlevel", "identifier", "source", "sxsi", "argomento", "unnamed"])

def search_philosophy_data(data):
    # https://en.wikipedia.org/wiki/Glossary_of_philosophy
    # https://en.wikipedia.org/wiki/List_of_philosophical_concepts
    # http://users.ox.ac.uk/~worc0337/philosophers.html
    for index,row in data.iterrows():
        pass

# pulisce il file al path tesi_UK_abs e toglie le righe senza abstract
# il dataset risultante è quello a tesi_UK_abs_mod
def clean_dataset(path = tesi_UK_abs):
    data = pd.read_csv(path,
                        sep="\t",
                        index_col=False,
                        skiprows=1,
                        names=["titolo","autore","univ","publisher","anno","abs","tipo","argomento","1","2","3","4","5","6"])
    print(data.count())
    dic = {"titolo":[],"autore":[],"univ":[],"publisher":[],"anno":[],"abs":[],"tipo":[],"argomento":[]}
    for index,row in data.iterrows():
        if str(row['abs'])!="nan":
            dic['titolo'].append(row['titolo'])
            dic['autore'].append(row['autore'])
            dic['univ'].append(row['univ'])
            dic['publisher'].append(row['publisher'])
            dic['anno'].append(row['anno'])
            dic['abs'].append(row['abs'])
            dic['tipo'].append(row['tipo'])
            dic['argomento'].append(row['argomento'])
    new_df = pd.DataFrame(dic)
    print(new_df.count())
    new_df.to_csv("df.csv", index=None, columns=["titolo","autore","univ","publisher","anno","abs","tipo","argomento"])

def add_identifier():
    data_abs = read_dataset_UK(True)
    data_no_abs = read_dataset_UK(False)
    ids_abs = [i for i in range(1,data_abs.count()[0]+1)]
    ids_no_abs = [i for i in range(data_abs.count()[0]+1,data_abs.count()[0]+data_no_abs.count()[0]+1)]
    print(len(ids_abs),"-",ids_abs[0], ids_abs[-1])
    print(len(ids_no_abs),"-",ids_no_abs[0], ids_no_abs[-1])
    data_abs.loc[:,"id"] = ids_abs
    data_no_abs.loc[:,"id"] = ids_no_abs
    data_abs.to_csv("test1.csv", index=None)
    data_no_abs.to_csv("test2.csv", index=None)
