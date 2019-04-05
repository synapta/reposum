import xml.etree.ElementTree as ET
import pandas as pd
import re

orcid_philosophy = "orcid_data/orcid_philosophy_sources.csv"
orcid_work = "{http://www.orcid.org/ns/work}"
orcid_common = "{http://www.orcid.org/ns/common}"

journal_blacklist = [
    "Philosophical Transactions of the Royal Society",
    "Philosophical Magazine"
]

philosophy_data = {
    "journal": [],
    "title": [],
    "abstract": []
}

def parse_XML(file_path):
     tree = ET.parse(file_path)
     root = tree.getroot()

     # journal is never none
     journal = root.find(orcid_work+"journal-title").text
     for jb in journal_blacklist:
         if re.match(jb, journal, re.IGNORECASE) is not None:
             return (None, None, None)
     title = root.find(orcid_work+"title").find(orcid_common+"title")
     if title is None:
         return
     description = root.find(orcid_work+"short-description") #may be None
     if description is None:
         return (journal, title.text, None)
     else:
         return (journal, title.text, description.text)

count = 0
for orcid_file in open(orcid_philosophy, "r"):
    print(count, end="\r")
    j, tit, desc = parse_XML(orcid_file[:-1])
    count += 1

    if j is None:
        continue
    else:
        philosophy_data['journal'].append(j)
        philosophy_data['title'].append(tit)
        philosophy_data['abstract'].append(desc)
print("")

df = pd.DataFrame(philosophy_data)
df.to_csv("orcid_data/orcid_philosophy.csv", index=None)
