from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
import pandas as pd
import nltk, re

debug = False

manual_stop_words = ['theory','chapter','dissertation','study','argue','argument','examine','discuss',
                        'present','concept','discussion','consider','conclude','work','second','section',
                        'thesis','conclusion','explore','introduce','discourse','subject','philosophical',
                        'analysis','problem','approach','philosophy','question','issue','view','provide',
                        'understand','attempt','interpretation','debate','important','offer','address',
                        'point','term','perspective','critical','concern','particular','role','method',
                        'notion','claim','do','fact','make','account','case','defend','position','objection',
                        'sense','certain','answer','solution','give','kind','version','explain','possible',
                        'object','response','common','reject','develop','general','focus','project',
                        'various','suggest']

def read_samples(phil_file, nphil_file):
    phils = remove_missing_abstract(pd.read_csv(phil_file))
    nphils = remove_missing_abstract(pd.read_csv(nphil_file))
    return [phils, nphils]

def remove_missing_abstract(df):
    return df[df['abstract'] != " dcterms_abstract:: @@MISSING-DATA"]

def select_missing_abstract(df):
    return df[df['abstract'] == " dcterms_abstract:: @@MISSING-DATA"]

def build_vectorizer(name, min_df=0.0, max_df=1.0, n_features=20000):
    if name == "cv":
        return CountVectorizer(stop_words=set(list(nltk.corpus.stopwords.words('english'))+manual_stop_words),
                                analyzer="word",
                                min_df=min_df,
                                max_df=max_df,
                                max_features=n_features)
    elif name == "tfidf":
        return TfidfVectorizer(stop_words=set(list(nltk.corpus.stopwords.words('english'))+manual_stop_words),
                                analyzer="word")

def lemmatize_data(dataframe, col_name, output=True):
    wnl = WordNetLemmatizer()
    res = []
    count = 0
    for index,row in dataframe.iterrows():
        if output:
            print(count, end="\r")
        count += 1
        words = [wnl.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else wnl.lemmatize(i) for i,j in pos_tag(word_tokenize(str(row[col_name])))]
        res.append(' '.join(w for w in words))
        if debug==True and count > 49:
            break
    if output:
        print("")
    return res

def preprocess_dataframe(dataframe, processer, check_length=True):
    dataframe = dataframe.applymap(lambda x: " ".join(s for s in processer(x)))
    dataframe = dataframe.applymap(lambda x: re.sub(r'[0-9]+',"",x))
    if check_length:
        dataframe = dataframe.applymap(lambda x: " ".join(s for s in x.split() if len(s) > 3))
    return dataframe

def save_lemmatized_data(phil_df, nphil_df, phil_file, nphil_file):
    phil_df.to_csv(phil_file, index=False)
    nphil_df.to_csv(nphil_file, index=False)

def read_lemmatized_data(phil_file, nphil_file):
    phils = pd.read_csv(phil_file)
    nphils = pd.read_csv(nphil_file)
    return [phils, nphils]
