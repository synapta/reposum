from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
from treetagger import TreeTagger
import pyLDAvis.sklearn
import pandas as pd

################################################################################

n_top_words = 30
n_features = 20000
n_topics = [20, 30,40,50]
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

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print()
        message = "Topic #%d: " % int(topic_idx+1)
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print("\n\n")

def visualize_lda(lda_model, TDmatrix, vectorizer, sort_t, path):
    panel = pyLDAvis.sklearn.prepare(lda_model, TDmatrix, vectorizer, mds='tsne', sort_topics=sort_t)
    pyLDAvis.save_html(panel, path)

################################################################################

cv = CountVectorizer(
    stop_words=set(list(stopwords)+manual_stopwords),
    analyzer="word",
    min_df=0.0,
    max_df=1.0,
    max_features=n_features
)

if not use_preprocessed_data:
    print("reading data...")
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

print("training vectorizer...")
cv.fit(df_id['preprocessed'])
TDmat = cv.transform(df_id['preprocessed'])
joblib.dump(cv, "models/cv_{}.pkl".format(n_features))

for num in n_topics:
    lda = LatentDirichletAllocation(
        n_components=num,
        max_iter=12,
        learning_method='online',
        learning_offset=30.,
        random_state=0,
        n_jobs=6
    )
    print("training lda with {} topics...".format(num))
    lda.fit(cv.transform(df_id['preprocessed']))
    print_top_words(lda,cv.get_feature_names(),10)

    joblib.dump(lda, "models/lda_{}_{}.pkl".format(num, n_features))
    visualize_lda(lda, TDmat, cv, True, "html/lda_{}_{}.html".format(num, n_features))
