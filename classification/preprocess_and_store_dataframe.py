from sklearn.externals import joblib
import text_processing as tp
import dataset_utils as dsu
import pandas as pd

vectorizer = tp.build_vectorizer("cv", n_features=None)
analyzer = vectorizer.build_analyzer()
"""
#data = dsu.read_dataset_UK_ethos(True)
data = pd.read_csv("data/no_philosophy_abs.csv")
print("data with abs size:",data.count()[0])
data_titles = pd.DataFrame(tp.lemmatize_data(data[['title']],"title"), columns=['title'])
data_titles = tp.preprocess_dataframe(data_titles, analyzer)
data_abs = pd.DataFrame(tp.lemmatize_data(data[['abstract']],"abstract"), columns=['abstract'])
data_abs = tp.preprocess_dataframe(data_abs, analyzer)
data_text = data_titles.merge(data_abs,left_index=True,right_index=True)

preprocessed_data = []
for index,row in data_text.iterrows():
    preprocessed_data.append(row['title'] + " " + row['abstract'])
print("preprocessed data size:",len(preprocessed_data))

data.loc[:,"preprocessed_data"] = preprocessed_data

data.to_csv("preprocessed_data/ethos_abs_preprocessed.csv",index=None, columns=["id","title","abstract","preprocessed_data"])
print("\n")

#data = dsu.read_dataset_UK_ethos(False)
data = pd.read_csv("data/no_philosophy_no_abs.csv")
print("data with no abs size:",data.count()[0])
data_titles = pd.DataFrame(tp.lemmatize_data(data[['title']],"title"), columns=['title'])
data_titles = tp.preprocess_dataframe(data_titles, analyzer)
data_text = data_titles

preprocessed_data = []
for index,row in data_text.iterrows():
    preprocessed_data.append(row['title'])
print("preprocessed data size:",len(preprocessed_data))

data.loc[:,"preprocessed_data"] = preprocessed_data

data.to_csv("preprocessed_data/ethos_no_abs_preprocessed.csv",index=None, columns=["id","title","preprocessed_data"])
print("\n")
"""

data = pd.read_csv("data/test_set_1000.tsv", delimiter="\t", names=['title','creator','university','publisher', 'year','abstract','type','subject','id','philosophy'])
print("test size:",data.count()[0])
data = data.fillna("")
test_titles = pd.DataFrame(tp.lemmatize_data(data[['title']],"title"), columns=['title'])
test_titles = tp.preprocess_dataframe(test_titles, analyzer)
test_abs = pd.DataFrame(tp.lemmatize_data(data[['abstract']],"abstract"), columns=['abstract'])
test_abs = tp.preprocess_dataframe(test_abs, analyzer)
data_text = test_titles.merge(test_abs,left_index=True,right_index=True)

preprocessed_data = []
for index,row in data_text.iterrows():
    preprocessed_data.append(row['title'] + " " + row['abstract'])
print("preprocessed data size:",len(preprocessed_data))

data_text.loc[:,"preprocessed_data"] = preprocessed_data
data_text.to_csv("preprocessed_data/test_1000_preprocessed.csv", index=None, columns=["title", "abstract", "preprocessed_data"])
print("\n")


data = dsu.read_orcid()
print("orcid data size:\n",data.count()[0])
dfa = data[data['abstract']!="no_abstract"]
dfn = data[data['abstract']=="no_abstract"]
for df_in,f_out,process_abs in [(dfa,"orcid_abs_preprocessed",True),(dfn,"orcid_no_abs_preprocessed",False)]:
    data_titles = pd.DataFrame(tp.lemmatize_data(df_in[['title']],"title"), columns=['title'])
    data_titles = tp.preprocess_dataframe(data_titles, analyzer)
    data_text = data_titles
    if process_abs:
        data_abs = pd.DataFrame(tp.lemmatize_data(df_in[['abstract']],"abstract"), columns=['abstract'])
        data_abs = tp.preprocess_dataframe(data_abs, analyzer)
        data_text = data_text.merge(data_abs,left_index=True,right_index=True)

    preprocessed_data = []
    for index,row in data_text.iterrows():
        if process_abs:
            row_text = row['title'] + " " + row['abstract']
        else:
            row_text = row['title']
        preprocessed_data.append(row_text)
    print("preprocessed data size:",len(preprocessed_data))

    df_in.loc[:,"preprocessed_data"] = preprocessed_data

    df_in.to_csv("preprocessed_data/"+f_out+".csv",index=None, columns=["title","abstract","preprocessed_data"])
print("\n")


data = dsu.read_doiboost()
print("doiboost data size:\n",data.count()[0])
dfa = data[data['abstract']!="no_abstract"]
dfn = data[data['abstract']=="no_abstract"]
for df_in,f_out,process_abs in [(dfa,"doiboost_abs_preprocessed", True),(dfn,"doiboost_no_abs_preprocessed",False)]:
    data_titles = pd.DataFrame(tp.lemmatize_data(df_in[['title']],"title"), columns=['title'])
    data_titles = tp.preprocess_dataframe(data_titles, analyzer)
    data_text = data_titles
    if process_abs:
        data_abs = pd.DataFrame(tp.lemmatize_data(df_in[['abstract']],"abstract"), columns=['abstract'])
        data_abs = tp.preprocess_dataframe(data_abs, analyzer)
        data_text = data_text.merge(data_abs,left_index=True,right_index=True)

    preprocessed_data = []
    for index,row in data_text.iterrows():
        if process_abs:
            row_text = row['title'] + " " + row['abstract']
        else:
            row_text = row['title']
        preprocessed_data.append(row_text)
    print("preprocessed data size:",len(preprocessed_data))

    df_in.loc[:,"preprocessed_data"] = preprocessed_data

    df_in.to_csv("preprocessed_data/"+f_out+".csv",index=None, columns=["title","abstract","preprocessed_data"])
print("\n")
