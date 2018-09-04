from timeout import TimeoutError
from timeout import timeout
import pandas as pd
import requests

url = "https://api.dandelion.eu/datatxt/nex/v1/"
token = "35776f47620f460bb2b8d83711959c46"

parameters = {
    "lang":"en",
    "token":token,
    "text":""
}

@timeout(2)
def sendRequest(url, heads=None):
    try:
        res = requests.get(url, headers=heads)
        return res
    except urllib.error.HTTPError:
        print("request error")

data = pd.read_csv("no_philosophy.csv")

for index, row in data.iterrows():
    text = ' '.join([row['title'], row['abstract']])
    parameters["text"] = text

    req_url = url + "?" + ''.join([k+"="+v+"&" for k,v in parameters.items()])

    while True:
        try:
            #time.sleep(0.001)
            resp = sendRequest(req_url[:-1])
            if resp != False and resp.status_code == 200:
                resp = resp.json()
            break
        except TimeoutError:
            print("timeout")
            continue

    print("number of entities:", len(resp['annotations']))
    for elem in resp['annotations']:
        print("\t",elem['spot'])
    input()
