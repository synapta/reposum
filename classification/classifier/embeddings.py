from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import euclidean, cosine
from anytree import Node, RenderTree, LevelOrderIter
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import mean_squared_error
from taxonomy import tax
import pandas as pd
import numpy as np
import re, nltk

debug = False
fasttext = "models/dataset.vec"
word2vec = "models/GoogleNews-vectors-negative300.bin"
numberbatch = "models/numberbatch-en-17.06.txt"
taxonomy_file = "data/Taxonomy_of_Philosophy.html"

tax_embeddings = {}
taxonomy = Node("philosophy")

##################################################################################################

def parseEmbeddings(path):
    data = {}
    dataset = open(path, "r", encoding="utf8")
    [num_words, embedding_size] = dataset.readline().split(" ")
    i=1
    print("Reading dataset file...")
    for line in dataset:
        [word, embedding] = line[:-1].split(" ",1)
        word = word.replace("_", " ")
        data[word] = embedding
        percentage = round( (100*(i/int(num_words))),1 )
        i += 1
        print("  %.1f%% complete" %percentage, end="\r")

        if percentage > 2.0 and debug: break

    print("\n")
    dataset.close()
    return data

def traverseDictEmbeds(d, level, preprocesser, embeds):
    if len(d.keys()) > 0:
        level_embeddings = []
        for key in d.keys():
            pkey = preprocesser(key)
            vec = transformText(embeds, ' '.join(pkey))
            if vec is not False:
                tax_embeddings[' '.join(pkey)] = vec[0]
                level_embeddings.append(vec[0])
                #print(' '.join(pkey))
                #print(level,len(level_embeddings),"\n")
            sublevel_aggregate_embeddings = traverseDictEmbeds(d[key], level+1, preprocesser, embeds)
            if sublevel_aggregate_embeddings is not None:
                level_embeddings.append(sublevel_aggregate_embeddings)
            #else:
                #input("dizionario vuoto")
        if level == 0:
            for elem in zip(level_embeddings, [node.name for node in LevelOrderIter(taxonomy, maxlevel=2)][1:]):
                tax_embeddings[elem[1]+str("_aggr")] = elem[0]
        else:
            #print("computing mean..................")
            #for arr in level_embeddings:
            #    input(arr)
            mean = np.mean(np.array(level_embeddings), axis=0)
            #input(mean)
            #if (np.isnan(mean)[0]==True):
            #    return None
            #else:
                #print("Aggiunto un elemento al livello precedente\n")
            return mean
    else:
        return None

def traverseDictTree(d, level, preprocesser, parent_node):
    if len(d.keys()) > 0:
        for key in d.keys():
            pkey = ' '.join(preprocesser(key))
            n = Node(pkey, parent=parent_node)
            traverseDictTree(d[key], level+1, preprocesser, n)

def traverseTaxonomyTree(vector):
    pass

def transformText(embed_dictionary, documents):
    string = False
    if type(documents) is str: #single document
        string = True
        documents = [documents]

    i = 0
    text_embeddings = []
    for document in documents:
        words = document.split()
        doc_embeddings = []
        for word in words:
            try:
                word_embedding = embed_dictionary[word]
                word_embedding = np.array(list(float(value) for value in word_embedding[:-1].split(" ")))
                doc_embeddings.append(word_embedding)
            except KeyError:
                continue
        if len(doc_embeddings) == 0:
            return False
        text_embeddings.append(np.mean(np.array(doc_embeddings),axis=0))
    return np.array(text_embeddings)

##################################################################################################

#WV = KeyedVectors.load_word2vec_format(word2vec, binary=True)
#NuB = parseEmbeddings(numberbatch)
FT = parseEmbeddings(fasttext) #fasttext wins !

cv = CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'))
analyzer = cv.build_analyzer()
"""
for embeds in [WV, NuB, FT]:
    tot_words = 0
    found_words = 0
    for line in open(taxonomy_file, "r"):
        words = analyzer(line[:-1])
        for w in words:
            tot_words += 1
            try:
                embeds[w]
                found_words += 1
            except KeyError:
                continue
    print("words:", str(found_words)+"/"+str(tot_words))
"""

traverseDictTree(tax, 0, analyzer, taxonomy) #build taxonomy tree
traverseDictEmbeds(tax, 0, analyzer, FT) #compute taxonomy embeddings

#read classification results
df = pd.read_csv("results/classification_abs_tfidf.csv")
for index,row in df[df['classification']==1].iterrows():
    #print thesis info
    print(index,row['argomento'],"\n",row['titolo'],"\n",row['abs'])
    print(row['prob_0'],row['prob_1'])
    input()

    vec = transformText(FT, row['argomento']+row['titolo'])[0]
    first_level_nodes = [node.name for node in LevelOrderIter(taxonomy, maxlevel=2)]
    input(first_level_nodes)
    errs = []
    dists = []
    cosines = []
    for node in first_level_nodes[1:]:
        embed = tax_embeddings[node]
        embed_aggr = tax_embeddings[node+"_aggr"]
        err = mean_squared_error(embed, vec)
        dist = euclidean(embed, vec)
        cos = cosine(embed, vec)
        errs.append(err)
        dists.append(dist)
        cosines.append(cos)
    print(np.mean(np.array(errs)), "\t\t\t", np.mean(np.array(dists)), "\t\t\t", np.mean(np.array(cosines)))
    errs = []
    dists = []
    cosines = []
    for node in first_level_nodes[1:]:
        embed_aggr = tax_embeddings[node+"_aggr"]
        err = mean_squared_error(embed_aggr, vec)
        dist = euclidean(embed_aggr, vec)
        cos = cosine(embed_aggr, vec)
        errs.append(err)
        dists.append(dist)
        cosines.append(cos)
    print(np.mean(np.array(errs)), "\t\t\t", np.mean(np.array(dists)), "\t\t\t", np.mean(np.array(cosines)))
    input()



























#
