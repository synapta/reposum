import pandas as pd
import spacy

fname = "../data/tesi_US/US_PhD_dissertations.xlsx"

print("Reading data...")
abstracts = pd.read_excel(fname, usecols=[24])

print("Preprocessing data...")
abstracts = abstracts[abstracts[' Abstract '] != "  Nessun elemento disponibile. "]
abstracts = abstracts[abstracts[' Abstract '] != "  Abstract Not Available. "]
abstracts = abstracts[abstracts[' Abstract '] != "  Abstract not available. "]
abstracts = abstracts[abstracts[' Abstract '] != "  Abstract Not Available "]
abstracts = abstracts[abstracts[' Abstract '] != "Abstract not available."]

nlp = spacy.load('en_core_web_sm')

for index,row in abstracts.iterrows():
    text = row[' Abstract ']
    print(text,"\n")
    doc = nlp(text)
    for ent in doc.ents:
        print("\t",ent.text, ent.label_)
    input("\n\n")
