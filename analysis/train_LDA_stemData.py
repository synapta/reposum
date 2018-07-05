from sklearn.feature_extraction.text import CountVectorizer
import lda, pickle, re, sys, pyLDAvis
from treetagger import TreeTagger
import pyLDAvis.sklearn
import pandas as pd
import numpy as np

num_topics = 30
n_features = 6000
n_top_words = 50

manual_stop_words = ['theory','chapter','dissertation','study','argue','argument','examine','discuss',
                        'present','concept','discussion','consider','conclude','work','second','section',
                        'thesis','conclusion','explore','introduce','discourse','subject','philosophical',
                        'analysis','problem','approach','philosophy','question','issue','view','provide',
                        'understand','attempt','interpretation','debate','important','offer','address',
                        'point','term','perspective','critical','concern','particular','role','method',
                        'notion']

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

fname = "../data/tesi_US/abstract_lemmas.csv"

eprint("Reading data...")
abstracts = pd.read_csv(fname, usecols=['abstract'])

cv = CountVectorizer(stop_words=manual_stop_words, analyzer="word", min_df=0.0, max_df=1.0)
analyzer = cv.build_analyzer()

eprint("Frequent words...")
top_freqs = get_top_n_words(abstracts['abstract'].tolist(), n_features)
words = list([word for word,freq in top_freqs])

eprint("Transformation...")
cv.fit(words)
TDmatrix = cv.transform(abstracts['abstract'].tolist())

eprint("LDA...")
model = lda.LDA(n_topics=num_topics, random_state=1)
model.fit_transform(TDmatrix)

topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(words)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    print("\n")

panel = pyLDAvis.sklearn.prepare(model, TDmatrix, cv, mds='tsne')
pyLDAvis.save_html(panel, 'HTMLtopics/LDA_'+str(fname.split("tesi_")[1][0:2])+'_'+str(num_topics)+'_'+str(n_features)+'.html')
