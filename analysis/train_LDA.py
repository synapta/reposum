from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import sys
import re

if len(sys.argv) < 2:
    print("Usage:",sys.argv[0],"<number of topics or list of topics>")
    sys.exit(1)
elif len(sys.argv) == 2:
    n_components = [int(sys.argv[1])]
else:
    n_components = list([int(sys.argv[i]) for i in range(1,len(sys.argv))])

fname = "../data/tesi_US/US_PhD_dissertations.xlsx"
n_features = [7500, 10000]

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

print("Reading data...")
abstracts = pd.read_excel(fname, usecols=[24])

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

for n_words in n_features:
    print("Selecting most frequent",n_words ,"words...")
    top_freqs = get_top_n_words(abstracts[' Abstract '], n_words)
    words = list([word for word,freq in top_freqs])

    print("   Computing term-document matrix...")
    TDmatrix = cv.fit(words)
    TDmatrix = cv.transform(feed_data(abstracts))

    for num in n_components:
        print("      starting LDA with",num,"topics...")
        lda = LatentDirichletAllocation(n_components=num,
                                        max_iter=15,
                                        learning_method='online',
                                        learning_offset=30.,
                                        random_state=0,
                                        n_jobs=-1)

        lda.fit(TDmatrix)
        joblib.dump(lda, "LDA_"+str(num)+"_"+str(n_words)[:-3]+"k.pkl")
