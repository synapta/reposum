from habanero import Crossref
import sys, re, os, time
import pandas as pd

batch_size = 50
cursor = "AoJ12Lv/2+gCPxFodHRwOi8vZHguZG9pLm9yZy8xMC4yODkyNS8yMjI2LTMwMTIuMjAxNy42LjcwNzU="
out_file = "crossref_philosophy.csv"

def query_crossref(cursor):
    return cr.works(
        query_title = "philosophy",
        cursor=cursor,
        cursor_max=batch_size,
        limit=batch_size,
        filter = {
            'has_abstract': True
        }
    )

def process_result(doc):
    id = doc['DOI']
    title = doc['title'][0]
    abstract = doc['abstract']
    abstract = re.sub(r'<[^>]*>', '', abstract)
    return (id, title, abstract)

def add_to_dict(entry):
    id, title, abstract = entry
    documents['doi'].append(id)
    documents['title'].append(title)
    documents['abstract'].append(abstract)

def save_data(last_cursor=False):
    if os.path.exists("crossref_philosophy.csv"):
        df_old = pd.read_csv("crossref_philosophy.csv")
        df = df_old.append(pd.DataFrame(documents))
    else:
        df = pd.DataFrame(documents)
    df.to_csv("crossref_philosophy.csv", index=None)
    if last_cursor:
        with open("last_cursor.txt", "w") as out_file:
            out_file.write(last_cursor)

######################################################

cr = Crossref()
documents = {
    'doi': [],
    'title': [],
    'abstract': []
}

print("querying...")
x = query_crossref(cursor)
print("done.")

for doc in x['message']['items']:
    res = process_result(doc)
    add_to_dict(res)

while len(x['message']['items']) >= batch_size:
    try:
        time.sleep(5)
        print("querying...")
        x = query_crossref(x['message']['next-cursor'])
        print("done.")

        for doc in x['message']['items']:
            res = process_result(doc)
            add_to_dict(res)

        print("Documents:", len(documents['doi']))
    except Exception as ex:
        print(ex)
        save_data(x['message']['next-cursor'])
        print("FINISHING WITH ERROR")
        sys.exit(1)

save_data()
print("FINISHING WITH NO ERROR")
sys.exit(0)
