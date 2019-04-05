import pandas as pd

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

def read_dataset_UK_ethos(abs=True):
    if abs:
        return pd.read_csv(tesi_UK_abs_ethos, delimiter="\t", skiprows=1, names=["id", "titolo","autore","univ","publisher","anno","abs","tipo","qname", "qlevel", "identifier", "source", "sxsi", "argomento", "unnamed"])
    else:
        return pd.read_csv(tesi_UK_nabs_ethos, delimiter="\t", skiprows=1, names=["id","titolo","autore","univ","publisher","anno","tipo","qname", "qlevel", "identifier", "source", "sxsi", "argomento", "unnamed"])

#journal
#title
#abstract
orcid_data = "../data_gathering/orcid_data/orcid_philosophy.csv"

def read_orcid():
    df = pd.read_csv(orcid_data)
    df = df.fillna("no_abstract")
    return df

#doi
#title
#abstract
doiboost_data = "../data_gathering/doiboost_data/doiboost_philosophy.csv"

def read_doiboost():
    df = pd.read_csv(doiboost_data)
    df = df.fillna("no_abstract")
    return df

#older functions
def read_samples(phil_file, nphil_file):
    phils = remove_missing_abstract(pd.read_csv(phil_file))
    nphils = remove_missing_abstract(pd.read_csv(nphil_file))
    return [phils, nphils]

def read_train_samples(phil_file, nphil_file):
    phils = pd.read_csv(phil_file)
    nphils = pd.read_csv(nphil_file)
    return [phils, nphils]

def remove_missing_abstract(df):
    return df[df['abstract'] != " dcterms_abstract:: @@MISSING-DATA"]

def select_missing_abstract(df):
    return df[df['abstract'] == " dcterms_abstract:: @@MISSING-DATA"]
