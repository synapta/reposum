from sklearn.externals import joblib
import text_processing as tp
import dataset_utils as dsu
import pandas as pd
import numpy as np

use_preprocessed = False

def feed_preprocessed_data(dataframe):
    for index,row in dataframe.iterrows():
        yield row['preprocessed_data']

def feed_data(dataframe, abs):
    for index,row in dataframe.iterrows():
        if abs:
            yield row['argomento']+' '+row['titolo']+' '+row['abs']
        else:
            yield row['argomento']+' '+row['titolo']

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

vectorizer = joblib.load("models/vectorizer_cv.pkl")
analyzer = vectorizer.build_analyzer()
clf = joblib.load("models/randomforestCLF_cv.pkl")

if use_preprocessed:
    data_abs = pd.read_csv("data/UK_abs_id.csv")
    data_no_abs = pd.read_csv("data/UK_no_abs_id.csv")
    print("Number of samples:",data_abs.count()[0], data_no_abs.count()[0])

    for data,label in zip([data_abs, data_no_abs],["abs", "no_abs"]):
        TDmatrix = vectorizer.transform(feed_preprocessed_data(data))
        res = clf.predict(TDmatrix)
        probs = clf.predict_proba(TDmatrix)
        res_df = add_columns_to_df(data, res, probs)
        res_df.to_csv("results/classification_"+label+".csv",index=None)
        print(label,"salvato")
else:
    #abstracts
    data = dsu.read_dataset_UK_id(True)
    data_subj = pd.DataFrame(tp.lemmatize_data(data[['argomento']],"argomento"), columns=['argomento'])
    data_subj = tp.preprocess_subjects(data_subj)
    data_titles = pd.DataFrame(tp.lemmatize_data(data[['titolo']],"titolo"), columns=['titolo'])
    data_titles = tp.preprocess_dataframe(data_titles, analyzer)
    data_abs = pd.DataFrame(tp.lemmatize_data(data[['abs']],"abs"), columns=['abs'])
    data_abs = tp.preprocess_dataframe(data_abs, analyzer)

    data_text = data_subj.merge(data_titles,left_index=True,right_index=True).merge(data_abs,left_index=True,right_index=True)
    print("samples:",data_text.count()[0])

    TDmatrix = vectorizer.transform(feed_data(data_text, True))
    res = clf.predict(TDmatrix)
    probs = clf.predict_proba(TDmatrix)
    res_df = add_columns_to_df(data, res, probs)
    res_df.to_csv("results/classification_abs.csv",index=None)
    print("abs salvato")

    #no abstracts
    data_no_abs = dsu.read_dataset_UK_id(False)
    data_subj = pd.DataFrame(tp.lemmatize_data(data[['argomento']],"argomento"), columns=['argomento'])
    data_subj = tp.preprocess_subjects(data_subj)
    data_titles = pd.DataFrame(tp.lemmatize_data(data[['titolo']],"titolo"), columns=['titolo'])
    data_titles = tp.preprocess_dataframe(data_titles, analyzer)

    data_text = data_subj.merge(data_titles,left_index=True,right_index=True)
    print("samples:",data_text.count()[0])

    TDmatrix = vectorizer.transform(feed_data(data_text, False))
    res = clf.predict(TDmatrix)
    probs = clf.predict_proba(TDmatrix)
    res_df = add_columns_to_df(data, res, probs)
    res_df.to_csv("results/classification_no_abs.csv",index=None)
    print("no abs salvato")
