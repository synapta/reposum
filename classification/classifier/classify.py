from sklearn.externals import joblib
import text_processing as tp
import dataset_utils as dsu
import pandas as pd

min_df = 0.0
max_df = 0.8
n_features = 60000

def feed_data():
    for index,row in data_text.iterrows():
        if str(row['abs'])!="nan":
            yield row['argomento']+' '+row['titolo']+' '+row['abs']

data = dsu.read_dataset_UK(False)
data = data.iloc[:100]
print("Number of samples:",data.count()[0])

cv = tp.build_vectorizer("cv", min_df, max_df, n_features)
analyzer = cv.build_analyzer()
vectorizer = joblib.load("vectorizer.pkl")

data_subj = pd.DataFrame(tp.lemmatize_data(data[['argomento']],"argomento"), columns=['argomento'])
data_titles = pd.DataFrame(tp.lemmatize_data(data[['titolo']],"titolo"), columns=['titolo'])
data_titles = tp.preprocess_dataframe(data_titles, analyzer)
data_abs = pd.DataFrame(tp.lemmatize_data(data[['abs']],"abs"), columns=['abs'])
data_abs = tp.preprocess_dataframe(data_abs, analyzer)

data_text = data_subj.merge(data_titles,left_index=True,right_index=True).merge(data_abs,left_index=True,right_index=True)

# transform text data into vector space
TDmatrix = vectorizer.transform(feed_data())
print("Vocabulary:",len(vectorizer.vocabulary_))
print("TDmatrix 1:",TDmatrix.shape)

clf = joblib.load("randomforestCLF.pkl")

res = clf.predict(TDmatrix)
acc_phil = np.mean(np.equal(res,1))
acc_nphil = np.mean(np.equal(res,0))
print("Philosophy:", acc_phil)
print("No philosophy:", acc_nphil)
