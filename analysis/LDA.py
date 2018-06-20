from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import pandas as pd

fname = "../data/tesi_US/US_PhD_dissertations.xlsx"
n_components = 20
n_features = 5000
n_top_words = 40
n_topics = [20, 30, 40, 50, 60, 70, 80, 90, 100]

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
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print("\n")

def savePlot(X, y, title, path):
    plt.clf()
    plt.plot(X, y, 'o-')
    plt.ylabel('perplexity')
    plt.xlabel('number of topics')
    plt.title(title)
    plt.savefig(path)

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

print("Selecting most frequent words...")
top_freqs = get_top_n_words(abstracts[' Abstract '], n_features)
words = list([word for word,freq in top_freqs])

print("Computing term-document matrix...")
TDmatrix = cv.fit(words)
TDmatrix = cv.transform(feed_data(abstracts))
#tf_feature_names = cv.get_feature_names()
#print_top_words(lda, tf_feature_names, n_top_words)

perplexities = []
for num in n_topics:
    print("LDA",num)
    lda = joblib.load("LDA_"+str(num)+"_"+str(n_features)[:-3]+"k.pkl")
    perp = lda.perplexity(TDmatrix)/num
    print("\tperplexity:", perp)
    perplexities.append(perp)

savePlot(n_topics, perplexities, "plot", "plot_"+str(n_features)[:-3]+"k.png")
