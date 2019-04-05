from spacy.util import minibatch, compounding
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from sklearn.externals import joblib
import psycopg2, nltk, pickle, re
import text_processing as tp
import thinc.extra.datasets
from pathlib import Path
import pandas as pd
import random
import spacy
import plac

vectorizer = joblib.load("models/vectorizer_mix.pkl")
analyzer = vectorizer.build_analyzer()

#train size
n_positive_samples = 50000
n_negative_samples = 500000
n_iter = 20

def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)

    tp = 0.0   # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0   # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]['cats']
        #print(doc)
        #print(gold)
        for label, score in doc.cats.items():
            #print(label)
            #print(score)
            """if label not in gold:
                input("caso 1")
                continue"""
            if score >= 0.5 and gold[label] >= 0.5:
                #input("caso 2")
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                #input("caso 3")
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                #input("caso 4")
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                #input("caso 5")
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    #f_score = 2 * (precision * recall) / (precision + recall)
    f_score = 0
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}

nlp_model = spacy.load('en')

# read and preprocess text data
test_df = pd.read_csv("data/test_set_1000.tsv", delimiter="\t", names=['title','creator','university','publisher', 'year','abstract','type','subject','id','philosophy'])
print("test size:",test_df.count()[0])
test_df = test_df.fillna("")
test_titles = pd.DataFrame(tp.lemmatize_data(test_df[['title']],"title"), columns=['title'])
test_titles = tp.preprocess_dataframe(test_titles, analyzer)
test_abs = pd.DataFrame(tp.lemmatize_data(test_df[['abstract']],"abstract"), columns=['abstract'])
test_abs = tp.preprocess_dataframe(test_abs, analyzer)
test_text = test_titles.merge(test_abs,left_index=True,right_index=True)

preprocessed_data = []
for index,row in test_text.iterrows():
    preprocessed_data.append(row['title'] + " " + row['abstract'])
print("preprocessed data size:",len(preprocessed_data))

test_df.loc[:,"preprocessed_data"] = preprocessed_data

orcid_abs = pd.read_csv("preprocessed_data/orcid_abs_preprocessed.csv")[['preprocessed_data']]
doiboost_abs = pd.read_csv("preprocessed_data/doiboost_abs_preprocessed.csv")[['preprocessed_data']]
data_abs = pd.concat([orcid_abs, doiboost_abs])
orcid_no_abs = pd.read_csv("preprocessed_data/orcid_no_abs_preprocessed.csv")[['preprocessed_data']]
doiboost_no_abs = pd.read_csv("preprocessed_data/doiboost_no_abs_preprocessed.csv")[['preprocessed_data']]
data_no_abs = pd.concat([orcid_no_abs, doiboost_no_abs])
phil_text = pd.concat([data_abs,data_no_abs])
#phil_text = data_abs
phil_text = phil_text.fillna("nan_value")
phil_text = phil_text[phil_text['preprocessed_data']!="nan_value"]

ethos_abs = pd.read_csv("preprocessed_data/ethos_abs_preprocessed.csv")[['preprocessed_data']]
ethos_no_abs = pd.read_csv("preprocessed_data/ethos_no_abs_preprocessed.csv")[['preprocessed_data']]
nphil_text = pd.concat([ethos_abs,ethos_no_abs])
#nphil_text = ethos_abs
nphil_text = nphil_text.fillna("nan_value")
nphil_text = nphil_text[nphil_text['preprocessed_data']!="nan_value"]

#shuffle dataframes and select a subset
phil_text = phil_text.sample(frac=1).iloc[0:n_positive_samples]
nphil_text = nphil_text.sample(frac=1).iloc[0:n_negative_samples]

textcat = nlp_model.create_pipe('textcat')
nlp_model.add_pipe(textcat, last=True)

textcat.add_label('POSITIVE')

train_data = list(zip( [row['preprocessed_data'] for index,row in phil_text.iterrows()],[{'cats': {'POSITIVE':True}} for _ in range(phil_text.count()[0])] ))
train_data.extend(list(zip( [row['preprocessed_data'] for index,row in nphil_text.iterrows()],[{'cats': {'POSITIVE':False}} for _ in range(nphil_text.count()[0])] )))

random.shuffle(train_data)

dev_texts = test_df['preprocessed_data'].tolist()
dev_cats = [{'cats': {'POSITIVE':True}} if row['philosophy']==1 else {'cats': {'POSITIVE':False}} for index,row in test_df.iterrows()]

other_pipes = [pipe for pipe in nlp_model.pipe_names if pipe != 'textcat']
with nlp_model.disable_pipes(*other_pipes):  # only train textcat
    optimizer = nlp_model.begin_training()
    print("Training the model...")
    print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
    for i in range(n_iter):
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(train_data, size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            #print(texts[0])
            #input(annotations[0])
            nlp_model.update(texts, annotations, sgd=optimizer, drop=0.2,
                       losses=losses)
        with textcat.model.use_params(optimizer.averages):
            # evaluate on the dev data split off in load_data()
            scores = evaluate(nlp_model.tokenizer, textcat, dev_texts, dev_cats)
        print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
              .format(losses['textcat'], scores['textcat_p'],
                      scores['textcat_r'], scores['textcat_f']))
