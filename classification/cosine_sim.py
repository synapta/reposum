from sklearn.metrics import recall_score, precision_score,f1_score
from sklearn.metrics.pairwise import cosine_similarity
import text_processing as tp
import pandas as pd
import numpy as np

master_N = 10000
n_positive_samples = master_N
n_negative_samples = master_N
n_evaluations = master_N

def feed_data_all():
    for index,row in phil_text.iterrows():
        if row['preprocessed_data'] != "nan_value":
            yield row['preprocessed_data']
    for index,row in nphil_text.iterrows():
        if row['preprocessed_data'] != "nan_value":
            yield row['preprocessed_data']

vectorizer = tp.build_vectorizer("cv", n_features = 200000)
analyzer = vectorizer.build_analyzer()

# read and preprocess text data
test_df = pd.read_csv("data/test_set_1000.tsv", delimiter="\t", names=['title','creator','university','publisher', 'year','abstract','type','subject','id','philosophy'])
test_df = test_df.fillna("nannn")
test_df = test_df[test_df['abstract']!="nannn"]
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

TDphilo = vectorizer.transform(phil_text['preprocessed_data'].tolist())
TDother = vectorizer.transform(nphil_text['preprocessed_data'].tolist())
TDtest = vectorizer.transform(test_df['preprocessed_data'].tolist())

y_true = np.array(test_df['philosophy'].tolist(), dtype=np.int64)
y_pred = []
for i in range(TDtest.shape[0]):
    print("evaluation: %s"%i, end="\r")

    test_vec = TDtest[i].A[0].reshape(1, -1)
    sim_philo = cosine_similarity(test_vec,TDphilo)[0][0]
    sim_other = cosine_similarity(test_vec,TDother)[0][0]

    m1 = np.mean(sim_philo)
    m2 = np.mean(sim_other)
    if m1 < m2:
        y_pred.append(1)
    else:
        y_pred.append(0)

print("\n")

y_pred = np.array(y_pred)
acc = str(np.mean(np.equal(y_true, y_pred)))[0:6]
precision = str(precision_score(y_true, y_pred))[0:6]
recall = str(recall_score(y_true, y_pred))[0:6]
f1score = str(f1_score(y_true, y_pred))[0:6]
print("ACCURACY\tPRECISION\tRECALL\tF1")
print("%s\t\t%s\t\t%s\t%s"%(acc,precision,recall,f1score))









#
