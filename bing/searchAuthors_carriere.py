from timeout import TimeoutError
from timeout import timeout
import pandas as pd
import urllib.error
import requests
import psycopg2
import time
import sys
import os
import re

url = 'https://api.cognitive.microsoft.com/bing/v7.0/search?mkt=it-IT&q='
headers = {'Ocp-Apim-Subscription-Key':'8006a3ee9ffa4faabae672b646c5ee2b',}

data_dir = "../data/"
tesi_US = ""
tesi_UK = "tesi_US/US_PhD_dissertations.xlsx"
carriere_dir = data_dir + "tesi_US/carriere/excel/"

@timeout(2)
def sendRequest(url, heads=None):
    try:
        res = requests.get(url, headers=heads)
        return res
    except urllib.error.HTTPError:
        print("request error")

with open("PQLaphrodite.txt", "r") as f:
    db = f.readline()[:-1]
    user = f.readline()[:-1]
    host = f.readline()[:-1]
    pwd = f.readline()[:-1]

try:
    connect_str = "dbname='"+db+"' user='"+user+"' host='"+host+"' " + \
                  "password='"+pwd+"'"
    conn = psycopg2.connect(connect_str)
    cur = conn.cursor()
except Exception as e:
    print(e)
    sys.exit(1)

data = pd.read_excel("excel/Wittgenstein in abstract 1981-2010.xlsx")

for index, row in data.iterrows():
    print(index)
    author = re.sub(r'^\s*',"",str(row[2]))
    if author == "nan":
        continue

    names = author.split()
    if len(names) > 2:
        author = names[1]+" "+names[0][0:-1]

    query1 = author
    query2 = "site:academia.edu+"+author
    query3 = "site:researchgate.net+"+author
    query4 = "site:linkedin.com+"+author
    for query in [query1, query2, query3, query4]:
        while True:
            try:
                time.sleep(0.001)
                resp = sendRequest(url+query, headers)
                if resp != False and resp.status_code == 200:
                    resp = resp.json()
                break
            except TimeoutError:
                print("timeout")
                continue

        try:
            test = resp["webPages"]
        except KeyError:
            print("\t0 results")
            break
        except TypeError:
            print("\tType error")
            continue

        for page in resp["webPages"]["value"]:
            try:
                cur.execute("INSERT INTO autori_bing (file, autore, query, url) VALUES (%s,%s,%s,%s);",("Wittgenstein in abstract 1981-2010.xlsx", author, query, page['url']))
                conn.commit()
            except psycopg2.ProgrammingError:
                print("Programming error")
                cur.execute("rollback")
            except psycopg2.DataError:
                print("Data error")
                cur.execute("rollback")
            except ValueError:
                print("Value error")
                cur.execute("rollback")
