from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import random, os

from sklearn.naive_bayes import GaussianNB, ComplementNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score,f1_score

useful_cols = {
    "anno": 1,
    #"abstract": 3,
    "uni": 4,
    "rank": 5,
    "soggetto": 6,
    "early-late": 7,
    "fascia": 11,
    "uni-arrivo": 13,
    "rank-arrivo": 14,
    #"nazione-uni": 15,
    "successo": 18
}
ucols = {
    "Dummett": {
        "uni": 6,
        "rank": 7,
        "uni-arrivo": 12,
        "rank-arrivo": 13,
        "successo": 14
    },
    "Wittgenstein": {
        "uni": 4,
        "rank": 5,
        "uni-arrivo": 13,
        "rank-arrivo": 14,
        "successo": 18
    },
    "Lewis": {
        "uni": 5,
        "rank": 6,
        "uni-arrivo": 11,
        "rank-arrivo": 12,
        "successo": 15
    },
    "Gadamer": {
        "uni": 6,
        "rank": 7,
        "uni-arrivo": 12,
        "rank-arrivo": 13,
        "successo": 14
    },
    "Kripke": {
        "uni": 6,
        "rank": 7,
        "uni-arrivo": 12,
        "rank-arrivo": 13,
        "successo": 15
    },
    "Fodor": {
        "uni": 6,
        "rank": 7,
        "uni-arrivo": 12,
        "rank-arrivo": 13,
        "successo": 14
    },
    "Spinoza": {
        "uni": 5,
        "rank": 6,
        "uni-arrivo": 12,
        "rank-arrivo": 13,
        "successo": 14
    }
}

le_rank = LabelEncoder()
le_succ = LabelEncoder()
le_phil = LabelEncoder()
le_uni = LabelEncoder()

train_data = []
train_labels = []

ranks = []
succs = []
phils = set()
unis = set()

base_dir = "../data/tesi_US/carriere/excel"
for file in os.listdir(base_dir):
    print("reading {}...".format(file))
    df = pd.read_excel(
        os.path.join(base_dir,file),
        usecols = ucols[file.split()[0]].values(),
        names = ucols[file.split()[0]].keys()
    )
    for col in ucols[file.split()[0]]:
        df = df[~df[col].isnull()]

    ranks.extend(list(df['rank']))
    ranks.extend(list(df['rank-arrivo']))
    succs.extend(list(df['successo']))
    phils.add(file.split()[0])
    for u in list(df['uni']):
        unis.add(u.strip().lower())
    for u in list(df['uni-arrivo']):
        unis.add(u.strip().lower())

le_rank.fit(list(set(ranks)))
le_succ.fit(list(set(succs)))
le_phil.fit(list(phils))
le_uni.fit(list(set(unis)))

for file in os.listdir(base_dir):
    df = pd.read_excel(
        os.path.join(base_dir,file),
        usecols = ucols[file.split()[0]].values(),
        names = ucols[file.split()[0]].keys()
    )
    for col in ucols[file.split()[0]]:
        df = df[~df[col].isnull()]

    for index, row in df.iterrows():
        train_data.append([
            #row['anno'],
            le_uni.transform([row['uni'].strip().lower()])[0],
            le_rank.transform([row['rank']])[0],
            #le_el.transform([row['early-late']])[0],
            le_uni.transform([row['uni-arrivo'].strip().lower()])[0],
            le_rank.transform([row['rank-arrivo']])[0],
            le_phil.transform([file.split()[0]])
        ])
        #train_labels.append(le_succ.transform([row['successo']])[0])
        train_labels.append(row['successo'])

seed = random.randint(0,1000)
X_train, X_test, y_train, y_test = train_test_split(
    train_data,
    train_labels,
    test_size = 0.2,
    random_state = seed
)

clf = LogisticRegression(
    solver='lbfgs',
    multi_class='multinomial',
    max_iter=500,
    n_jobs=6
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

y_true = y_test

for i in range(len(y_true)):
    print(str(y_true[i])+" "+str(y_pred[i]))

acc = str(accuracy_score(y_true, y_pred))[0:6]
#precision = str(precision_score(y_true, y_pred, average='micro'))
#precision = str(precision_score(y_true, y_pred, average='macro'))
#precision = str(precision_score(y_true, y_pred, average='weighted'))
precision = str(precision_score(y_true, y_pred, average=None))
#precision = str(precision_score(y_true, y_pred, average='samples'))
print("Showing results for seed {}".format(seed))
print("ACCURACY")
print(acc)
print("PRECISION")
print(precision)
