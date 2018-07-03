from sklearn.feature_extraction.text import CountVectorizer
from treetagger import TreeTagger
import lda, pickle, re, sys
import pandas as pd
import numpy as np

num_topics = 10
n_top_words = 100
n_words = 6000

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def lemmatize_data(data):
    tt = TreeTagger(language='english')
    res = []
    count = 0
    for index,row in data.iterrows():
        count += 1
        eprint(count)
        abstract = []
        for elem in tt.tag(row[' Abstract ']):
            lemma = elem[2]
            if lemma == '<unknown>':
                abstract.append(elem[0])
            elif len(lemma.split("|")) == 2:
                parts = lemma.split("|")
                if len(parts[0]) < len(parts[1]):
                    abstract.append(parts[0])
                else:
                    abstract.append(parts[1])
            else:
                abstract.append(lemma)
        if len(abstract) > 0:
            res.append(' '.join(word for word in abstract))
    return res

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

fname = "../data/tesi_US/US_PhD_dissertations.xlsx"

eprint("Reading data...")
abstracts = pd.read_excel(fname, usecols=[24])

eprint("Preprocessing data...")
abstracts = abstracts[abstracts[' Abstract '] != "  Nessun elemento disponibile. "]
abstracts = abstracts[abstracts[' Abstract '] != "  Abstract Not Available. "]
abstracts = abstracts[abstracts[' Abstract '] != "  Abstract not available. "]
abstracts = abstracts[abstracts[' Abstract '] != "  Abstract Not Available "]
abstracts = abstracts[abstracts[' Abstract '] != "Abstract not available."]
abstracts = abstracts[abstracts[' Abstract '] != " EMPTY "]

cv = CountVectorizer(stop_words="english", analyzer="word", min_df=0.0, max_df=1.0)
analyzer = cv.build_analyzer()

abstracts = abstracts.applymap(lambda x: " ".join(s for s in analyzer(x)))
abstracts = abstracts.applymap(lambda x: re.sub(r'[0-9]+',"",x))
abstracts = abstracts.applymap(lambda x: " ".join(s for s in x.split() if len(s) > 3))
eprint("Lemmatization...")
train_data = lemmatize_data(abstracts)

eprint("Frequent words...")
top_freqs = get_top_n_words(train_data, n_words)
words = list([word for word,freq in top_freqs])

eprint("Transformation...")
cv.fit(words)
TDmatrix = cv.transform(train_data)

eprint("LDA...")
model = lda.LDA(n_topics=num_topics, random_state=1)
model.fit_transform(TDmatrix)

with open("output/LDAmodel.pkl","wb") as f:
    pickle.dump(model,f)

topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(words)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    print("\n")
