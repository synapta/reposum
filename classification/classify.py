from sklearn.externals import joblib
import text_processing as tp
import dataset_utils as dsu
import pandas as pd
import numpy as np

use_preprocessed = True

def feed_preprocessed_data(dataframe):
    for index,row in dataframe.iterrows():
        yield row['preprocessed_data']

def feed_data(dataframe, abs):
    for index,row in dataframe.iterrows():
        if abs:
            yield row['titolo']
        else:
            yield row['titolo']

def add_columns_to_df(df, clf_res, clf_probs):
    prob0 = []
    prob1 = []
    for elem in clf_probs:
        prob0.append(elem[0])
        prob1.append(elem[1])
    df.loc[:,"classification"] = clf_res
    df.loc[:,"prob_0"] = prob0
    df.loc[:,"prob_1"] = prob1
    return df

vectorizer = joblib.load("models/vectorizer.pkl")
analyzer = vectorizer.build_analyzer()
clf = joblib.load("models/randomforestCLF.pkl")

if use_preprocessed:
    data_abs = pd.read_csv("preprocessed_data/ethos_abs_preprocessed.csv")
    data_no_abs = pd.read_csv("preprocessed_data/ethos_no_abs_preprocessed.csv")
    print("Number of samples:",data_abs.count()[0], data_no_abs.count()[0])

    for data,label in zip([data_abs, data_no_abs],["abs", "no_abs"]):
        TDmatrix = vectorizer.transform(feed_preprocessed_data(data))
        res = clf.predict(TDmatrix)
        probs = clf.predict_proba(TDmatrix)
        res_df = add_columns_to_df(data, res, probs)
        res_df.to_csv("results/classification_"+label+".csv",index=None)
        print(label,"saved")
else:
    #abstracts
    data = dsu.read_dataset_UK_ethos(True)
    data_titles = pd.DataFrame(tp.lemmatize_data(data[['titolo']],"titolo"), columns=['titolo'])
    data_titles = tp.preprocess_dataframe(data_titles, analyzer)
    data_text = data_titles
    print("samples:",data_text.count()[0])

    TDmatrix = vectorizer.transform(feed_data(data_text, True))
    res = clf.predict(TDmatrix)
    probs = clf.predict_proba(TDmatrix)
    res_df = add_columns_to_df(data, res, probs)
    res_df.to_csv("results/classification_abs.csv",index=None)
    print("abs saved")

    #no abstracts
    data_no_abs = dsu.read_dataset_UK_id(False)
    data_titles = pd.DataFrame(tp.lemmatize_data(data[['titolo']],"titolo"), columns=['titolo'])
    data_titles = tp.preprocess_dataframe(data_titles, analyzer)
    data_text = data_titles
    print("samples:",data_text.count()[0])

    TDmatrix = vectorizer.transform(feed_data(data_text, False))
    res = clf.predict(TDmatrix)
    probs = clf.predict_proba(TDmatrix)
    res_df = add_columns_to_df(data, res, probs)
    res_df.to_csv("results/classification_no_abs.csv",index=None)
    print("no abs saved")
