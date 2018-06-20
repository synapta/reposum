from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np
import re

fname = "../data/tesi_US/US_PhD_dissertations.xlsx"
n_components = 50
n_top_words = 40

def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.

    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) ->
    [('python', 2),
     ('world', 2),
     ('love', 2),
     ('hello', 1),
     ('is', 1),
     ('programming', 1),
     ('the', 1),
     ('language', 1)]
    """
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
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

print("Reading data...")
abstracts = pd.read_excel(fname, usecols=[24])
#abstracts = pd.read_excel(fname)

print("Preprocessing data...")
abstracts = abstracts[abstracts[' Abstract '] != "  Nessun elemento disponibile. "]
abstracts = abstracts[abstracts[' Abstract '] != "  Abstract Not Available. "]
abstracts = abstracts[abstracts[' Abstract '] != "  Abstract not available. "]
abstracts = abstracts[abstracts[' Abstract '] != "  Abstract Not Available "]
abstracts = abstracts[abstracts[' Abstract '] != "Abstract not available."]

cv = CountVectorizer(stop_words="english", analyzer="word")
analyzer = cv.build_analyzer()

abstracts = abstracts.applymap(lambda x:analyzer(x))
abstracts = abstracts.applymap(lambda x: " ".join(s for s in x))

print("Selecting most frequent words...")
top_freqs = get_top_n_words(abstracts[' Abstract '], 5000)
words = list([word for word,freq in top_freqs])

print("Computing term-document matrix...")
TDmatrix = cv.fit(words)
TDmatrix = cv.transform(feed_data(abstracts))

print("starting LDA...")
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0,
                                n_jobs=-1)
lda.fit(TDmatrix)
tf_feature_names = cv.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
