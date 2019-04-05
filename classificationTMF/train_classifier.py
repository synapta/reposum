from sklearn.metrics import recall_score, precision_score,f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn_pandas import DataFrameMapper
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from random import shuffle
from sklearn import svm
import pandas as pd
import numpy as np
import pickle

features_title = 10000
features_abstract = 20000
need_to_select_data = False
compute_importances = False
prune_features = True
save_models = False

ents_file = "../tmf_entities/tmf_entities_scores_UK.csv"
freq_files = [
    "../tmf_entities/entities_freq_title.tsv",
    "../tmf_entities/entities_freq_abstract.tsv"
]

def my_tokenizer(txt):
    return txt.lower().split()

def prepare_data(useful_ids):
    data = pd.read_csv(
        ents_file,
        delimiter="\t",
        names=['id', 'src', 'entity', 'score']
    )
    data = data[data['id'].isin(useful_ids)]
    last_id = data.iloc[0]['id']
    text = {"id": [], "title": [], "abstract": []}
    t_temp = ""
    a_temp = ""
    for index,row in data.iterrows():
        if index%1000 == 0:
            print(index)
        if row['id'] != last_id:
            text['id'].append(last_id)
            text['title'].append(t_temp)
            text['abstract'].append(a_temp)
            last_id = int(row['id'])
            t_temp = ""
            a_temp = ""
        if row['src'] == 'title':
            t_temp = t_temp + " " + str(row['entity'])
        else:
            a_temp = a_temp + " " + str(row['entity'])
    text['id'].append(int(last_id))
    text['title'].append(t_temp)
    text['abstract'].append(a_temp)
    return pd.DataFrame(text)

def read_samples(phil_file, nphil_file):
    phils = remove_missing_abstract(pd.read_csv(phil_file))
    nphils = remove_missing_abstract(pd.read_csv(nphil_file))
    return phils, nphils

def remove_missing_abstract(df):
    return df[df['abstract'] != " dcterms_abstract:: @@MISSING-DATA"]

df_title = pd.read_csv(freq_files[0], delimiter="\t", names=['entity','freq'])
df_abstract = pd.read_csv(freq_files[1], delimiter="\t", names=['entity','freq'])
ents_title = list(df_title['entity'])[0:features_title]
ents_abstract = list(df_abstract['entity'])[0:features_abstract]
if prune_features:
    importances_df = pd.read_csv("importances.csv", delimiter="\t")
    importances_df = importances_df[(importances_df['std']/2) > importances_df['importance']]
    bad_features = list(importances_df['name'])
    ents_title = [e for e in ents_title if e not in bad_features]
    ents_abstract = [e for e in ents_abstract if e not in bad_features]
effective_features = len(ents_title) + len(ents_abstract)

positive_samples_train, negative_samples_train = read_samples(
    "data/new_philosophy_train.csv",
    "data/nophilosophy_train.csv"
)
test_samples = pd.read_csv(
    "data/test_set_1000.tsv",
    delimiter="\t",
    names=[
        'title','creator','university','publisher', 'year','abstract','type',
        'subject','id','philosophy'
    ]
)

if need_to_select_data:
    id_pos = list(positive_samples_train['id'])
    id_neg = list(negative_samples_train['id'])
    id_test = list(test_samples['id'])
    data_df = prepare_data(id_pos+id_neg+id_test)

    #train data
    data_train_pos = data_df[data_df['id'].isin(id_pos)]
    data_train_pos = data_train_pos.copy()
    data_train_pos = data_train_pos.fillna("")
    data_train_neg = data_df[data_df['id'].isin(id_neg)]
    data_train_neg = data_train_neg.copy()
    data_train_neg = data_train_neg.fillna("")
    data_train = data_train_pos.append(data_train_neg)
    labels_train = [1 for _ in range(len(id_pos))] + [0 for _ in range(len(id_neg))]
    data_train.loc[:,"philosophy"] = labels_train

    #test data
    data_test = data_df[data_df['id'].isin(id_test)]
    data_test.fillna("")
    labels_test = list(test_samples['philosophy'])
    data_test.loc[:,"philosophy"] = labels_test

    data_train.to_csv("data/data_train.csv", index=None)
    data_test.to_csv("data/data_test.csv", index=None)
else:
    data_train = pd.read_csv("data/data_train.csv")
    data_train = data_train.fillna("")
    labels_train = list(data_train['philosophy'])
    data_train.drop(['philosophy'], axis=1)
    data_test = pd.read_csv("data/data_test.csv")
    data_test = data_test.fillna("")
    labels_test = list(data_test['philosophy'])
    data_test.drop(['philosophy'], axis=1)

cv_title = CountVectorizer(
    vocabulary = ents_title,
    stop_words = None,
    tokenizer = my_tokenizer,
    preprocessor = None,
)
cv_abstract = CountVectorizer(
    vocabulary = ents_abstract,
    stop_words = None,
    tokenizer = my_tokenizer,
    preprocessor = None,
)

tuple_array = [
    ('title', cv_title),
    ('abstract', cv_abstract),
]
mapper = DataFrameMapper(tuple_array, sparse=True)
mapper.fit(data_train.append(data_test))
matrix_train = mapper.transform(data_train)
matrix_test = mapper.transform(data_test)
if save_models:
    pickle.dump(mapper, open("models/mapper_{}.pkl".format(effective_features), "wb"))

indices = np.arange(matrix_train.shape[0])
shuffle(indices)
matrix_train = matrix_train[list(indices)]
labels_train = np.array(labels_train)[indices]

clf = RandomForestClassifier(
    max_depth=None,
    random_state=None,
    n_estimators=100,
    max_features=0.75,
    n_jobs=6,
    verbose=2
)
clf.fit(matrix_train, labels_train)
if save_models:
    pickle.dump(clf, open("models/clf_{}.pkl".format(effective_features), "wb"))

y_true = labels_test
y_pred = clf.predict(matrix_test)

print(list(zip(y_true,y_pred)))

acc = str(np.mean(np.equal(y_true, y_pred)))[0:6]
precision = str(precision_score(y_true, y_pred))[0:6]
recall = str(recall_score(y_true, y_pred))[0:6]
f1score = str(f1_score(y_true, y_pred))[0:6]
print("ACCURACY\tPRECISION\tRECALL\tF1")
print("%s\t\t%s\t\t%s\t%s"%(acc,precision,recall,f1score))

if compute_importances:
    model = clf
    num_features = effective_features
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
    indices = np.argsort(importances)[::-1][0:num_features]
    feature_names = cv_abstract.get_feature_names() + cv_title.get_feature_names()

    with open("importances.csv", "w") as f:
        f.write("name\timportance\tstd\n")
        for ind in indices:
            f.write(str(feature_names[ind])+"\t"+str(importances[ind])+"\t"+str(std[ind])+"\n")
