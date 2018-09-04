from sklearn.externals import joblib
import text_processing as tp
import dataset_utils as dsu
import pandas as pd
import numpy as np

def feed_data():
    for index,row in data_text.iterrows():
        if str(row['abs'])!="nan":
            yield row['argomento']+' '+row['titolo']+' '+row['abs']

data = dsu.read_dataset_UK(False).iloc[:5000]
#data = data.iloc[:1000]
print("Number of samples:",data.count()[0])

vectorizer = joblib.load("vectorizer.pkl")
analyzer = vectorizer.build_analyzer()

data_subj = pd.DataFrame(tp.lemmatize_data(data[['argomento']],"argomento"), columns=['argomento'])
data_titles = pd.DataFrame(tp.lemmatize_data(data[['titolo']],"titolo"), columns=['titolo'])
data_titles = tp.preprocess_dataframe(data_titles, analyzer)
data_abs = pd.DataFrame(tp.lemmatize_data(data[['abs']],"abs"), columns=['abs'])
data_abs = tp.preprocess_dataframe(data_abs, analyzer)

data_text = data_subj.merge(data_titles,left_index=True,right_index=True).merge(data_abs,left_index=True,right_index=True)

print("data_text:",data_text.count())

# transform text data into vector space
TDmatrix = vectorizer.transform(feed_data())
print("Vocabulary:",len(vectorizer.vocabulary_))
print("TDmatrix 1:",TDmatrix.shape)

clf = joblib.load("randomforestCLF.pkl")

res = clf.predict(TDmatrix)
probs = clf.predict_proba(TDmatrix)
acc_phil = np.mean(np.equal(res,1))
acc_nphil = np.mean(np.equal(res,0))
print("Philosophy:", acc_phil)
print("No philosophy:", acc_nphil)

for i in range(len(res)):
    if probs[i][1] > 0.3:
        print(probs[i])
        print(data.iloc[i]["titolo"])
        print(data.iloc[i]["argomento"])
        input()
