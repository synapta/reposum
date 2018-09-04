from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import euclidean
from sklearn.externals import joblib
import text_processing as tp
from random import randint
import pandas as pd
import numpy as np
import sys
from sklearn.svm import SVC, LinearSVC

use_preprocessed = True
negative_ratio = 4
min_df = 0.0
max_df = 0.8
n_features = 60000
test_ratio = 0.1

def feed_data():
    for index,row in phil_text.iterrows():
        if str(row['abstract'])!="nan":
            yield row['subject']+' '+row['title']+' '+row['abstract']
    for index,row in nphil_text.iterrows():
        if str(row['abstract'])!="nan":
            yield row['subject']+' '+row['title']+' '+row['abstract']

def feed_data1():
    for index,row in phil_text.iterrows():
        if str(row['abstract'])!="nan":
            yield row['subject']+' '+row['title']+' '+row['abstract']

def feed_data2():
    for index,row in nphil_text.iterrows():
        if str(row['abstract'])!="nan":
            yield row['subject']+' '+row['title']+' '+row['abstract']

def feed_classifier():
    yield TDmatrix1
    yield TDmatrix2

cv = tp.build_vectorizer("cv", min_df, max_df, n_features)
analyzer = cv.build_analyzer()
tfidf = tp.build_vectorizer("tfidf")
vectorizer = tfidf
#vectorizer = joblib.load("vectorizer.pkl")

# preprocess text data
if not use_preprocessed:
    # read text data
    [positive_samples, negative_samples] = tp.read_samples("philosophy.csv", "no_philosophy.csv")
    num = negative_samples.count()[0] - 1
    rand_indexes = [randint(0,num) for _ in range(len(positive_samples)*negative_ratio)]
    negative_samples = negative_samples.iloc[rand_indexes]

    print("Positive samples:",positive_samples.count()[0])
    print("Negative samples:",negative_samples.count()[0])

    phil_subj = pd.DataFrame(tp.lemmatize_data(positive_samples[['subject']],"subject"), columns=['subject'])
    phil_titles = pd.DataFrame(tp.lemmatize_data(positive_samples[['title']],"title"), columns=['title'])
    phil_titles = tp.preprocess_dataframe(phil_titles, analyzer)
    phil_abs = pd.DataFrame(tp.lemmatize_data(positive_samples[['abstract']],"abstract"), columns=['abstract'])
    phil_abs = tp.preprocess_dataframe(phil_abs, analyzer)

    nphil_subj = pd.DataFrame(tp.lemmatize_data(negative_samples[['subject']],"subject"), columns=['subject'])
    nphil_titles = pd.DataFrame(tp.lemmatize_data(negative_samples[['title']],"title"), columns=['title'])
    nphil_titles = tp.preprocess_dataframe(nphil_titles, analyzer)
    nphil_abs = pd.DataFrame(tp.lemmatize_data(negative_samples[['abstract']],"abstract"), columns=['abstract'])
    nphil_abs = tp.preprocess_dataframe(nphil_abs, analyzer)

    phil_text = phil_subj.merge(phil_titles,left_index=True,right_index=True).merge(phil_abs,left_index=True,right_index=True)
    nphil_text = nphil_subj.merge(nphil_titles,left_index=True,right_index=True).merge(nphil_abs,left_index=True,right_index=True)
    tp.save_lemmatized_data(phil_text, nphil_text, "preprocessed_data/phil_text.csv", "preprocessed_data/nphil_text.csv")
else:
    [phil_text, nphil_text] = tp.read_lemmatized_data("preprocessed_data/phil_text.csv", "preprocessed_data/nphil_text.csv")
    if nphil_text.count()[0] > phil_text.count()[0]*negative_ratio:
        num = nphil_text.count()[0] - 1
        rand_indexes = [randint(0,num) for _ in range(phil_text.count()[0]*negative_ratio)]
        nphil_text = nphil_text.iloc[rand_indexes]
    elif nphil_text.count()[0] < phil_text.count()[0]*negative_ratio:
        print("Error. Not enough negative samples stored in nphil_abs.csv")
        sys.exit(1)

# transform text data into vector space
vectorizer.fit(feed_data())
joblib.dump(vectorizer, "vectorizer.pkl")
TDmatrix1 = vectorizer.transform(feed_data1())
TDmatrix2 = vectorizer.transform(feed_data2())
print("Vocabulary:",len(vectorizer.vocabulary_))
print("TDmatrix 1:",TDmatrix1.shape)
print("TDmatrix 2:",TDmatrix2.shape)

train_data = []
test_data_pos = []
test_data_neg = []
labels = []
threshold1 = TDmatrix1.shape[0]-int(TDmatrix1.shape[0]*test_ratio)
threshold2 = TDmatrix2.shape[0]-int(TDmatrix2.shape[0]*test_ratio)
"""for i in range(TDmatrix1.shape[0]):
    if i < threshold1:
        train_data.append(TDmatrix1[i].A[0])
        labels.append(1)
    else:
        test_data_pos.append(TDmatrix1[i].A[0])
for i in range(TDmatrix2.shape[0]):
    if i < threshold2:
        train_data.append(TDmatrix2[i].A[0])
        labels.append(0)
    else:
        test_data_neg.append(TDmatrix2[i].A[0])
"""
for i in range(TDmatrix1.shape[0]):
    train_data.append(TDmatrix1[i].A[0])
    labels.append(1)
for i in range(TDmatrix2.shape[0]):
    train_data.append(TDmatrix2[i].A[0])
    labels.append(0)

s = np.arange(len(train_data))
np.random.shuffle(s)
train_data = np.array(train_data)[s]
labels = np.array(labels)[s]

# classificazione
"""clf = RandomForestClassifier(max_depth=None,
                                random_state=None,
                                n_estimators=40,
                                max_features=0.75,
                                n_jobs=6,
                                verbose=2)"""
clf = LinearSVC(verbose=2)
clf.fit(train_data[:-500],labels[:-500])
#res1 = clf.predict(test_data_pos)
#res2 = clf.predict(test_data_neg)
#acc1 = np.mean(np.equal(res1,1))
#acc2 = np.mean(np.equal(res2,0))
#print("Pos accuracy:",acc1)
#print("Neg accuracy:",acc2)
res = clf.predict(train_data[-500:])
acc = np.mean(np.equal(res,labels[-500:]))
print("acc:",acc)

joblib.dump(clf, "LinearSVC_CLF.pkl")

"""
# distanza di vettori
for i in range(TDmatrix1.shape[0]):
    sample = TDmatrix1[i].A[0]
    dist1 = []
    dist2 = []
    for j in range(TDmatrix1.shape[0]):
        vec = TDmatrix1[j].A[0]
        if j != i:
            dist1.append(euclidean(vec,sample))
    for j in range(TDmatrix2.shape[0]):
        vec = TDmatrix2[j].A[0]
        dist2.append(euclidean(vec,sample))
    print("")
    print("Vector",i)
    print("Dist 1:",np.mean(np.array(dist1)))
    print("Dist 2:",np.mean(np.array(dist2)))
    print("")

for i in range(TDmatrix2.shape[0]):
    sample = TDmatrix2[i].A[0]
    dist1 = []
    dist2 = []
    for j in range(TDmatrix2.shape[0]):
        vec = TDmatrix2[j].A[0]
        if j != i:
            dist2.append(euclidean(vec,sample))
    for j in range(TDmatrix1.shape[0]):
        vec = TDmatrix1[j].A[0]
        dist1.append(euclidean(vec,sample))
    print("")
    print("Vector",i)
    print("Dist 1:",np.mean(np.array(dist1)))
    print("Dist 2:",np.mean(np.array(dist2)))
    print("")
"""
