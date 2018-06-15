import pandas as pd
import spacy

tesi_US = "./data/tesi_US/US_PhD_dissertations.xlsx"

data = pd.read_excel(tesi_US)

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
    print(row[0])
    print(row[1])
    print(row[2])
    print(row[3])
    input()
