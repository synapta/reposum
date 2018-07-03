from nltk import word_tokenize, pos_tag, ne_chunk
from spacy.pipeline import EntityRecognizer
from rake_nltk import Rake
import pandas as pd
import spacy

tesi_US = "../data/tesi_US/US_PhD_dissertations.xlsx"

data = pd.read_excel(tesi_US, skiprows=9000, nrows=1000)

nlp = spacy.load('en_core_web_sm')
ner = EntityRecognizer(nlp.vocab)
rake = Rake(max_length=1)

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
for index,row in data.iterrows():
    abstract = row[24]
    doc = nlp(abstract)
    #processed = ner(doc)
    #input(processed)
    print(abstract, end="\n\n")
    print("[spacy]\t",end="")
    for entity in doc.ents:
        print(entity.text, entity.label_, end="  -  ")

    print("\n")

    ner = ne_chunk(pos_tag(word_tokenize(abstract)), binary=True)
    print("[nltk]\t",end="")
    for part in ner:
        if hasattr(part, "label"):
            print(" ".join(ent[0] for ent in part), end="  -  ")

    print("\n")

    rake.extract_keywords_from_text(abstract)
    kws = rake.get_ranked_phrases_with_scores()
    print("[rake]\t",kws)

    input("\n\n")
