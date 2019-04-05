from sklearn.metrics import recall_score, precision_score,f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import text_processing as tp
from sklearn.svm import SVC
from random import randint
import pandas as pd
import numpy as np
import sys, math

#train size
n_positive_samples = 50000
n_negative_samples = 200000
#vectorizer options
min_df = 0.0
max_df = 0.6
n_features = 80000
# RF params
max_depth_p = [None]
n_estimators_p = [10]
max_features_p = [0.95]
min_samples_leaf_p = [1]

def feed_data_all():
    for index,row in phil_text.iterrows():
        if row['preprocessed_data'] != "nan_value":
            yield row['preprocessed_data']
    for index,row in nphil_text.iterrows():
        if row['preprocessed_data'] != "nan_value":
            yield row['preprocessed_data']

def feed_data(df):
    for index,row in df.iterrows():
        if row['preprocessed_data'] != "nan_value":
            yield row['preprocessed_data']

def print_sample(sample, philo):
    print(sample)
    print("----------------------------------")
    print("PHILOSOPHY:%s"%philo)
    input("==================================")

def feed_data_vec(df1, df2, indexes):
    size1 = df1.count()[0]
    for index in indexes:
        if index >= size1:
            #print_sample(df2.iloc[index-size1]['preprocessed_data'], False)
            yield df2.iloc[index-size1]['preprocessed_data']
        else:
            #print_sample(df1.iloc[index]['preprocessed_data'], True)
            yield df1.iloc[index]['preprocessed_data']


cv = tp.build_vectorizer("tfidf", min_df, max_df, n_features)
analyzer = cv.build_analyzer()
vectorizer = cv

# read and preprocess text data
test_df = pd.read_csv("data/test_set_1000.tsv", delimiter="\t", names=['title','creator','university','publisher', 'year','abstract','type','subject','id','philosophy'])
print("test size:",test_df.count()[0])
test_df = test_df.fillna("")
test_titles = pd.DataFrame(tp.lemmatize_data(test_df[['title']],"title"), columns=['title'])
test_titles = tp.preprocess_dataframe(test_titles, analyzer)
test_abs = pd.DataFrame(tp.lemmatize_data(test_df[['abstract']],"abstract"), columns=['abstract'])
test_abs = tp.preprocess_dataframe(test_abs, analyzer)
test_text = test_titles.merge(test_abs,left_index=True,right_index=True)

preprocessed_data = []
for index,row in test_text.iterrows():
    preprocessed_data.append(row['title'] + " " + row['abstract'])
print("preprocessed data size:",len(preprocessed_data))

test_df.loc[:,"preprocessed_data"] = preprocessed_data

orcid_abs = pd.read_csv("preprocessed_data/orcid_abs_preprocessed.csv")[['preprocessed_data']]
doiboost_abs = pd.read_csv("preprocessed_data/doiboost_abs_preprocessed.csv")[['preprocessed_data']]
data_abs = pd.concat([orcid_abs, doiboost_abs])
orcid_no_abs = pd.read_csv("preprocessed_data/orcid_no_abs_preprocessed.csv")[['preprocessed_data']]
doiboost_no_abs = pd.read_csv("preprocessed_data/doiboost_no_abs_preprocessed.csv")[['preprocessed_data']]
data_no_abs = pd.concat([orcid_no_abs, doiboost_no_abs])
phil_text = pd.concat([data_abs,data_no_abs])
#phil_text = data_abs
phil_text = phil_text.fillna("nan_value")
phil_text = phil_text[phil_text['preprocessed_data']!="nan_value"]

ethos_abs = pd.read_csv("preprocessed_data/ethos_abs_preprocessed.csv")[['preprocessed_data']]
ethos_no_abs = pd.read_csv("preprocessed_data/ethos_no_abs_preprocessed.csv")[['preprocessed_data']]
nphil_text = pd.concat([ethos_abs,ethos_no_abs])
#nphil_text = ethos_abs
nphil_text = nphil_text.fillna("nan_value")
nphil_text = nphil_text[nphil_text['preprocessed_data']!="nan_value"]

#shuffle dataframes and select a subset
phil_text = phil_text.sample(frac=1).iloc[0:n_positive_samples]
nphil_text = nphil_text.sample(frac=1).iloc[0:n_negative_samples]

# transform text data into vector space
vectorizer.fit(feed_data_all())
print("vectorizer trained")
joblib.dump(vectorizer, "models/vectorizer_mix.pkl")

s1 = phil_text.count()[0]
s2 = nphil_text.count()[0]
s = np.arange(s1 + s2)
labels = [True if elem < s1 else False for elem in s]
np.random.shuffle(s)
labels = np.array(labels)[s]

TDmatrix = vectorizer.transform(feed_data_vec(phil_text, nphil_text, s))
TDmatrix_test = vectorizer.transform(feed_data(test_df))
print("Vocabulary:",len(vectorizer.vocabulary_))
print("TDmatrix:",TDmatrix.shape)
print("TDmatrix test:",TDmatrix_test.shape)

for max_depth in max_depth_p:
    for max_features in max_features_p:
        for n_estimators in n_estimators_p:
            for min_samples_leaf in min_samples_leaf_p:

                """clf = RandomForestClassifier(
                    max_depth=max_depth,
                    random_state=None,
                    n_estimators=n_estimators,
                    max_features=max_features,
                    n_jobs=6,
                    verbose=1
                )"""

                clf = SVC(
                    verbose=2
                )

                #train classifier
                clf.fit(TDmatrix, labels)
                clf.verbose = 0

                print("max_depth: %s"%max_depth)
                print("max_features: %s"%max_features)
                print("n_estimators: %s"%n_estimators)
                print("min_samples_leaf: %s"%min_samples_leaf)
                #evaluate on training set
                print("SET\tACCURACY\tPRECISION\tRECALL\tF1")
                y_true = labels
                y_pred = clf.predict(TDmatrix)
                acc = str(np.mean(np.equal(y_true, y_pred)))[0:6]
                precision = str(precision_score(y_true, y_pred))[0:6]
                recall = str(recall_score(y_true, y_pred))[0:6]
                f1score = str(f1_score(y_true, y_pred))[0:6]
                print("train\t%s\t\t%s\t\t%s\t%s"%(acc,precision,recall,f1score))

                #evaluate on test set
                y_true = np.array(test_df['philosophy'].tolist(), dtype=np.int64)
                y_pred = clf.predict(TDmatrix_test)
                acc = str(np.mean(np.equal(y_true, y_pred)))[0:6]
                precision = str(precision_score(y_true, y_pred))[0:6]
                recall = str(recall_score(y_true, y_pred))[0:6]
                f1score = str(f1_score(y_true, y_pred))[0:6]
                print("test\t%s\t\t%s\t\t%s\t%s"%(acc,precision,recall,f1score))

                print("\n")

joblib.dump(clf, "models/randomforestCLF_tuning.pkl")
