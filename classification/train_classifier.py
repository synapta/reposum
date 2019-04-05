from sklearn.metrics import recall_score, precision_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import text_processing as tp
from random import randint
import pandas as pd
import numpy as np
import sys, math

#train options
cross_validation = False
#train data options
min_df = 0.0
max_df = 0.8
n_features = 60000

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

def feed_data_vec(df1, df2, indexes):
    size1 = df1.count()[0]
    for index in indexes:
        if index >= size1:
            yield df2.iloc[index-size1]['preprocessed_data']
        else:
            yield df1.iloc[index]['preprocessed_data']


cv = tp.build_vectorizer("cv", min_df, max_df, n_features)
analyzer = cv.build_analyzer()
#vectorizer = tp.build_vectorizer("tfidf")
vectorizer = cv

# read and preprocess text data
#test_text = pd.read_csv("preprocessed_data/test_1000_preprocessed.csv")[['preprocessed_data']]
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
phil_text = phil_text.fillna("nan_value")
phil_text = phil_text[phil_text['preprocessed_data']!="nan_value"]

ethos_abs = pd.read_csv("preprocessed_data/ethos_abs_preprocessed.csv")[['preprocessed_data']]
ethos_no_abs = pd.read_csv("preprocessed_data/ethos_no_abs_preprocessed.csv")[['preprocessed_data']]
nphil_text = pd.concat([ethos_abs,ethos_no_abs])
nphil_text = nphil_text.fillna("nan_value")
nphil_text = nphil_text[nphil_text['preprocessed_data']!="nan_value"]

#shuffle dataframes and select a subset
phil_text = phil_text.sample(frac=1).iloc[0:5000]
nphil_text = nphil_text.sample(frac=1).iloc[0:50000]

# transform text data into vector space
vectorizer.fit(feed_data_all())
print("vectorizer trained")
joblib.dump(vectorizer, "models/vectorizer_mix.pkl")

s1 = phil_text.count()[0]
s2 = nphil_text.count()[0]
s = np.arange(s1 + s2)
labels = [True if elem < s1 else False for elem in s ]
np.random.shuffle(s)
labels = np.array(labels)[s]

TDmatrix = vectorizer.transform(feed_data_vec(phil_text, nphil_text, s))

"""
TDmatrix1 = vectorizer.transform(feed_data(phil_text))
print("Philosophy text transformed")
TDmatrix2 = vectorizer.transform(feed_data(nphil_text))
print("Non-philosophy text transformed")
"""
TDmatrix_test = vectorizer.transform(feed_data(test_df))
print("Test data transformed")
print("Vocabulary:",len(vectorizer.vocabulary_))
print("TDmatrix:",TDmatrix.shape)
#print("TDmatrix 2:",TDmatrix2.shape)
print("TDmatrix test:",TDmatrix_test.shape)

"""train_data = []
test_data = []
labels = []
labels_test = []
for i in range(TDmatrix1.shape[0]):
    train_data.append(TDmatrix1[i].A[0])
    labels.append(1)
for i in range(TDmatrix2.shape[0]):
    train_data.append(TDmatrix2[i].A[0])
    labels.append(0)
for i in range(TDmatrix_test.shape[0]):
    test_data.append(TDmatrix_test[i].A[0])
    labels_test.append(test_df.iloc[i]['philosophy'])"""

"""
#shuffle data and labels
s = np.arange(len(train_data))
np.random.shuffle(s)
train_data = np.array(train_data)[s]
labels = np.array(labels)[s]
"""
# classifier
clf = RandomForestClassifier(max_depth=None,
                                random_state=None,
                                n_estimators=100,
                                max_features=0.6,
                                n_jobs=6, #WARNING: consider changing it
                                verbose=2)

if cross_validation:
    #10-fold cross-validation
    k_fold = KFold(n_splits=10)
    it = 1
    accuracies = []
    precisions = []
    recalls = []
    f1scores = []
    for train, test in k_fold.split(train_data):
        y_true = labels[test]
        print("cv iteration",it,"...")
        clf.fit(train_data[train], labels[train])
        y_pred = clf.predict(train_data[test])
        acc = np.mean(np.equal(y_pred,labels[test]))
        accuracies.append(acc)
        precisions.append(precision_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))
        f1scores.append(f1_score(y_true, y_pred))

        print("accuracy at iteration",it,":",acc)
        print("precision at iteration",it,":",precision_score(y_true, y_pred))
        print("recall at iteration",it,":",recall_score(y_true, y_pred))
        print("f1 at iteration",it,":",f1_score(y_true, y_pred))
        it += 1
    print("\n\ncross-validation accuracy:",np.mean(np.array(accuracies)))
    print("cross-validation precision:",np.mean(np.array(precisions)))
    print("cross-validation recall:",np.mean(np.array(recalls)))
    print("cross-validation f1:",np.mean(np.array(f1scores)))
print("\n\n")

#train classifier
clf.fit(TDmatrix, labels)
clf.verbose = 0

#evaluate on training set
print("SET\tACCURACY\tPRECISION\tRECALL\tF1")
y_true = labels
y_pred = clf.predict(TDmatrix)
acc = str(np.mean(np.equal(y_true, y_pred)))[0:6]
precision = str(precision_score(y_true, y_pred))[0:6]
recall = str(recall_score(y_true, y_pred))[0:6]
f1score = str(f1_score(y_true, y_pred))[0:6]
print("train\t%s\t%s\t%s\t%s"%(acc,precision,recall,f1score))

#evaluate on test set
y_true = np.array(test_df['philosophy'].tolist(), dtype=np.int64)
y_pred = clf.predict(TDmatrix_test)
acc = str(np.mean(np.equal(y_true, y_pred)))[0:6]
precision = str(precision_score(y_true, y_pred))[0:6]
recall = str(recall_score(y_true, y_pred))[0:6]
f1score = str(f1_score(y_true, y_pred))[0:6]
#print("\n\ntest set Accuracy:",acc)
#print("test set Precision:",precision)
#print("test set Recall:",recall)
#print("test set F1:",f1score)
print("test\t%s\t%s\t%s\t%s"%(acc,precision,recall,f1score))

#joblib.dump(clf, "models/randomforestCLF_mix.pkl")
