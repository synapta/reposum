import pandas as pd
import os, re, json

root_dir = '/mnt/data/reposum/doiboost/doiBoost/'
out_file = 'doiboost_philosophy.csv'

documents = {
'doi': [],
'title': [],
'abstract': []
}

def check_philosophy(field_name):
    if re.search(r'[P|p]hilosop',field_name) is not None:
        return True
    else:
        return False

def get_longest_abstract(abstracts):
    max_len = 0
    max_len_idx = None
    for index,abs in enumerate(abstracts):
        abs_len = len(abs['value'])
        if abs_len > max_len:
            max_len = abs_len
            max_len_idx = index
    return abstracts[max_len_idx]['value']

def process_document(document):
    doi = doc['doi']
    title = doc['title']
    if len(title) == 0:
        return
    else:
        title = title[0]

    abstract = doc['abstract']
    if len(abstract) == 0:
        abstract = None
    elif len(abstract) == 1:
        abstract = abstract[0]['value']
    elif len(abstract) > 1:
        abstract = get_longest_abstract(abstract)

    add_to_dict(doi, title, abstract)

def add_to_dict(id, title, abstract):
    documents['doi'].append(id)
    documents['title'].append(title)
    documents['abstract'].append(abstract)

def save_dataframe():
    df = pd.DataFrame(documents)
    df.to_csv(out_file, index=None)

for file in os.listdir(root_dir):
    file_path = os.path.join(root_dir, file)

    print(file_path)
    count = 0

    for line in open(file_path, "r"):
        if line[-1] == '\n':
            line = line[1:-2]
        else:
            line = line[1:-1]
        try:
            doc = eval(line)
        except SyntaxError:
            continue

        subjects = doc['subject']
        for s in subjects:
            if check_philosophy(s):
                process_document(doc)

save_dataframe()
