from sklearn.feature_extraction.text import CountVectorizer
from sklearn_pandas import DataFrameMapper
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from random import shuffle
import pandas as pd
import numpy as np

features_title = 2000
features_abstract = 10000
need_to_select_data = False
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
    data_train.to_csv("data/data_train.csv", index=None)
else:
    data_train = pd.read_csv("data/data_train.csv")
    data_train = data_train.fillna("")
    labels_train = list(data_train['philosophy'])
    data_train.drop(['philosophy'], axis=1)

    #for test
    data_train = data_train.iloc[:12000]
    labels_train = labels_train[:12000]

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
matrix_train = mapper.fit_transform(data_train)

pca = PCA(n_components=3)
matrix_red = pca.fit_transform(matrix_train.todense())

fig = plt.figure()
ax = plt.axes(projection='3d')
#plt.scatter(matrix_red[:,0][:np.sum(labels_train)], matrix_red[:,1][:np.sum(labels_train)], color='red')
#plt.scatter(matrix_red[:,0][np.sum(labels_train):], matrix_red[:,1][np.sum(labels_train):], color='blue')
ax.scatter(
    matrix_red[:,0][:np.sum(labels_train)],
    matrix_red[:,1][:np.sum(labels_train)],
    matrix_red[:,2][:np.sum(labels_train)],
    color='red'
)
ax.scatter(
    matrix_red[:,0][np.sum(labels_train):],
    matrix_red[:,1][np.sum(labels_train):],
    matrix_red[:,2][np.sum(labels_train):],
    color='blue'
)
plt.show()
