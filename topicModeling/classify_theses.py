from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
from treetagger import TreeTagger
import pandas as pd
import heapq

################################################################################

n_features = 20000
n_topics = [20,30,40,50]
n_top_topics = 10
use_preprocessed_data = True

manual_stopwords = [
    'theory', 'work', 'chapter', 'majority', 'study', 'dissertation', 'claim',
    'philosophy', 'argue', 'argument', 'approach', 'research', 'student',
    'participant', 'data', 'read', 'literature', 'write', 'text', 'literary',
    'object', 'change', 'project', 'view', 'problem', 'concept', 'discourse',
    'test', 'report', 'result', 'hypothesis'
]

################################################################################

def lemmatize_data(data):
    tt = TreeTagger(language='english')
    lemm_data = {"abstract": []}
    count = 0
    for index,row in data.iterrows():
        print(count)
        count += 1
        abstract = ''
        if len(row['abstract']) != 0:
            for word, _, lemma in tt.tag(row['abstract']):
                if lemma == '<unknown>':
                    abstract += word + ' '
                elif "|" in lemma:
                    parts = lemma.split("|")
                    abstract += min(parts, key=len) + ' '
                else:
                    abstract += lemma + ' '
        lemm_data['abstract'].append(abstract)
    return pd.DataFrame(lemm_data)

################################################################################

cv = joblib.load("models/cv_{}.pkl".format(n_features))

print("reading data...")
if not use_preprocessed_data:
    df_id = pd.read_excel(
        "../data/tesi_US/US_PhD_dissertations.xlsx",
        usecols=[13,24],
        names=['id','abstract']
    )
    df_id = df_id.applymap(lambda x: str(x).strip())
    df_id = df_id[df_id['abstract'] != 'Abstract Not Available.']
    df_id = df_id[df_id['abstract'] != 'Abstract Not available.']
    df_id = df_id[df_id['abstract'] != 'Abstract not available.']
    df_id = df_id[df_id['abstract'] != 'abstract not available.']
    df_id = df_id[df_id['abstract'] != 'Nessun elemento disponibile.']

    analyzer = cv.build_analyzer()
    tt = TreeTagger(language='english')

    print("preprocessing...")
    df_id = df_id.applymap(lambda x: " ".join(word for word in analyzer(x)))
    df_id = df_id.applymap(lambda x: " ".join(word for word in x.split() if len(word) > 2))
    df_id = df_id[df_id['abstract'] != '']
    df_id.loc[:,'preprocessed'] = list(lemmatize_data(df_id)['abstract'])
    df_id = df_id.drop(['abstract'], axis=1)
    df_id.to_csv("data/tesi_US_preprocessed.csv", index=None)
else:
    df_id = pd.read_csv("data/tesi_US_preprocessed.csv")

ids = list(df_id['id'])
for num in n_topics:
    print("generating topics with model: lda_{}_{}".format(num, n_features))
    lda = joblib.load("models/lda_{}_{}.pkl".format(num, n_features))
    probs = lda.transform(cv.transform(df_id['preprocessed']))

    print("saving topics...")
    with open("out/probs_{}_{}.csv".format(num, n_features), "w") as outfile:
        outfile.write("id,topic,prob\n")
        for id, t_probs in zip(ids, probs):
            highest_probs = heapq.nlargest(n_top_topics, t_probs)
            for prob in sorted(highest_probs, reverse=True):
                ind = list(t_probs).index(prob)
                outfile.write(str(id)+","+str(ind)+","+str(t_probs[ind])[0:6]+"\n")
