from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import sys
import re
import os

fname = "../data/tesi_US/US_PhD_dissertations.xlsx"
lda_model = "models/LDA_s_40_6k.pkl"
n_top_words = 30
n_topics = [40]
n_features = 6000

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def feed_data(input_data):
    for index,row in input_data.iterrows():
        yield row[' Abstract ']

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print()
        message = "Topic #%d: " % int(topic_idx+1)
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print("\n\n")

print("Reading data...")
abstracts = pd.read_excel(fname, usecols=[24])

print("Preprocessing data...")
abstracts = abstracts[abstracts[' Abstract '] != "  Nessun elemento disponibile. "]
abstracts = abstracts[abstracts[' Abstract '] != "  Abstract Not Available. "]
abstracts = abstracts[abstracts[' Abstract '] != "  Abstract not available. "]
abstracts = abstracts[abstracts[' Abstract '] != "  Abstract Not Available "]
abstracts = abstracts[abstracts[' Abstract '] != "Abstract not available."]

cv = CountVectorizer(stop_words="english", analyzer="word", max_df=0.7)
analyzer = cv.build_analyzer()

abstracts = abstracts.applymap(lambda x: " ".join(s for s in analyzer(x)))
if lda_model.split("_")[1] == "s":
    from spacy.lang.en.lemmatizer.lookup import LOOKUP
    abstracts = abstracts.applymap(lambda x: " ".join(LOOKUP[s] for s in x.split() if s in LOOKUP))

print("Selecting most frequent",n_features,"words...")
top_freqs = get_top_n_words(abstracts[' Abstract '], n_features)
words = list([word for word,freq in top_freqs])

print("   Computing term-document matrix...")
TDmatrix = cv.fit(words)
TDmatrix = cv.transform(feed_data(abstracts))
tf_feature_names = cv.get_feature_names()

for num in n_topics:
    print("      LDA",num)
    lda = joblib.load(lda_model)
    print_top_words(lda, tf_feature_names, n_top_words)
