# prendere un po' di sample da phil_text e nphil_text, trasformarli
# e classificarli come in co-occurrence.py e vedere se il risultato
# della classificazione è giusto

from sklearn.externals import joblib
import text_processing as tp
from random import randint
import pandas as pd
import re

data = tp.remove_missing_abstract(pd.read_csv("no_philosophy.csv"))
#data = tp.select_missing_abstract(pd.read_csv("no_philosophy.csv"))

vectorizer = joblib.load("vectorizer.pkl")
analyzer = vectorizer.build_analyzer()
#clf = joblib.load("randomforestCLF.pkl")
clf = joblib.load("LinearSVC_CLF.pkl")

while True:
    index = randint(0,min(data.count()))
    row = data.iloc[[index]]

    abstract_df = pd.DataFrame(tp.lemmatize_data(row,"abstract",False), columns=['abstract'])
    abstract = tp.preprocess_dataframe(abstract_df, analyzer).iloc[0]["abstract"]

    title_df = pd.DataFrame(tp.lemmatize_data(row,"title",False), columns=['title'])
    title = tp.preprocess_dataframe(title_df, analyzer).iloc[0]["title"]

    subject_df = pd.DataFrame(tp.lemmatize_data(row,"subject",False), columns=['subject'])
    subject = re.sub(r'[\.\,\(\)\[\]\;\']', '', subject_df.iloc[0]["subject"])

    text = ' '.join([subject,title,abstract])
    vec = vectorizer.transform([text])
    res = clf.predict(vec)
    #probs = clf.predict_proba(vec)
    if res[0] == 1:
        print("\n")
        print(res[0])
        #print(probs)
        print(row.iloc[0]['subject'])
        print(row.iloc[0]['title'])
        print(row.iloc[0]['abstract'])
        input()

while True:
    index = randint(0,min(data.count()))
    row = data.iloc[[index]]

    title_df = pd.DataFrame(tp.lemmatize_data(row,"title",False), columns=['title'])
    title = tp.preprocess_dataframe(title_df, analyzer).iloc[0]["title"]

    subject_df = pd.DataFrame(tp.lemmatize_data(row,"subject",False), columns=['subject'])
    subject = re.sub(r'[\.\,\(\)\[\]\;\']', '', subject_df.iloc[0]["subject"])

    text = ' '.join([subject,title])
    vec = vectorizer.transform([text])
    res = clf.predict(vec)
    #probs = clf.predict_proba(vec)
    if res[0] == 1:
        print("\n")
        print(res[0])
        #print(probs)
        print(row.iloc[0]['subject'])
        print(row.iloc[0]['title'])
        input()
