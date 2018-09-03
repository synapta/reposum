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
tesi_US = "../../data/tesi_US/US_PhD_dissertations.xlsx"

#dc:title
#dc:creator
#uketdterms:institution
#dc:publisher
#dcterms:issued
#dcterms:abstract
#uketdterms:qualificationname
#dc:subject
tesi_UK_abs = "../../data/tesi_UK/tab_separated_value/171215/full_data/171215_1137_Synapta_EThOS_with_abstract.tsv"
tesi_UK_nabs = "../../data/tesi_UK/tab_separated_value/171215/full_data/171215_1137_Synapta_EThOS_NO_abstract.tsv"

def read_dataset_US():
    return pd.read_csv(tesi_US)

def read_dataset_UK(abs=True):
    if abs:
        return pd.read_csv(tesi_UK_abs,
                            sep="\t",
                            index_col=False,
                            names=["titolo","autore","univ","publisher","anno","abs","tipo","argomento","1","2","3","4","5","6"])
    else:
        return pd.read_csv(tesi_UK_nabs,
                            sep="\t",
                            index_col=False,
                            header=0,
                            names=["titolo","autore","univ","publisher","anno","abs","tipo","argomento"])

def search_philosophy_data(data):
    # https://en.wikipedia.org/wiki/Glossary_of_philosophy
    # https://en.wikipedia.org/wiki/List_of_philosophical_concepts
    # http://users.ox.ac.uk/~worc0337/philosophers.html
    for index,row in data.iterrows():
        pass
