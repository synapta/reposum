from sklearn.externals import joblib
import pyLDAvis.sklearn
import pandas as pd
import pickle, os

def visualize_lda(lda_model, TDmatrix, vectorizer, sort_t, path):
    panel = pyLDAvis.sklearn.prepare(lda_model, TDmatrix, vectorizer, mds='tsne', sort_topics=sort_t)
    pyLDAvis.save_html(panel, path)

df_id = pd.read_csv("data/tesi_US_preprocessed.csv")

for file in os.listdir("html/"):
    lda_file = os.path.join("html", file)
    n_topics = int(lda_file.split("_")[1])
    n_features = int(lda_file.split("_")[2].split(".")[0])

    cv = pickle.load(open("models/cv_{}.pkl".format(n_features), "rb"))
    cv.fit(df_id['preprocessed'])
    TDmat = cv.transform(df_id['preprocessed'])
    lda = joblib.load("models/lda_{}_{}.pkl".format(n_topics, n_features))
    visualize_lda(lda, TDmat, cv, False, "html/lda_{}_{}.html".format(n_topics, n_features))
