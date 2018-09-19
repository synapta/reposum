from sklearn.externals import joblib
import text_processing as tp
import dataset_utils as dsu
import pandas as pd

vectorizer = joblib.load("models/vectorizer.pkl")
analyzer = vectorizer.build_analyzer()

data = dsu.read_dataset_UK_id(True)
print("data 1 size:",data.count())
data_subj = pd.DataFrame(tp.lemmatize_data(data[['argomento']],"argomento"), columns=['argomento'])
data_subj = tp.preprocess_subjects(data_subj)
data_titles = pd.DataFrame(tp.lemmatize_data(data[['titolo']],"titolo"), columns=['titolo'])
data_titles = tp.preprocess_dataframe(data_titles, analyzer)
data_abs = pd.DataFrame(tp.lemmatize_data(data[['abs']],"abs"), columns=['abs'])
data_abs = tp.preprocess_dataframe(data_abs, analyzer)
data_text = data_subj.merge(data_titles,left_index=True,right_index=True).merge(data_abs,left_index=True,right_index=True)

preprocessed_data = []
for index,row in data.iterrows():
    preprocessed_data.append(row['argomento']+' '+row['titolo']+' '+row['abs'])
print("preprocessed size:",len(preprocessed_data))

data.loc[:,"preprocessed_data"] = preprocessed_data

data.to_csv("data/UK_abs_id.csv",index=None, columns=["id","titolo","autore","univ","publisher","anno","abs","tipo","argomento","preprocessed_data"])


data = dsu.read_dataset_UK_id(False)
print("data 2 size:",data.count())
data.ix[data['abs']!='nan']
data_subj = pd.DataFrame(tp.lemmatize_data(data[['argomento']],"argomento"), columns=['argomento'])
data_subj = tp.preprocess_subjects(data_subj)
data_titles = pd.DataFrame(tp.lemmatize_data(data[['titolo']],"titolo"), columns=['titolo'])
data_titles = tp.preprocess_dataframe(data_titles, analyzer)
data_text = data_subj.merge(data_titles,left_index=True,right_index=True)

preprocessed_data = []
for index,row in data_text.iterrows():
    preprocessed_data.append(row['argomento']+' '+row['titolo'])
print("preprocessed size:",len(preprocessed_data))

data.loc[:,"preprocessed_data"] = preprocessed_data

data.to_csv("data/UK_no_abs_id.csv",index=None, columns=["id","titolo","autore","univ","publisher","anno","tipo","argomento","preprocessed_data"])
