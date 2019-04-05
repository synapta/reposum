import xml.etree.ElementTree as ET
import pandas as pd
import re, os

root_dir = "/mnt/data/reposum/orcid/activities/"
out_dir = "orcid_data/"
out_file = "orcid_data/orcid_philosophy.csv"
orcid_work = "{http://www.orcid.org/ns/work}"
orcid_common = "{http://www.orcid.org/ns/common}"

def check_philosophy(journal_name):
    if re.search(r'[P|p]hilosop',journal_name) is not None:
        return True
    else:
        return False

def parse_XML(file_path):
     tree = ET.parse(file_path)
     root = tree.getroot()

     journal = root.find(orcid_work+"journal-title")
     if journal == None:
         return False

     #title = root.find(orcid_work+"title").find(orcid_common+"title")
     #if title is None:
     #    return
     #description = root.find(orcid_work+"short-description") #may be None

     #if check_philosophy(journal.text):
     #  print(file_path)
     try:
         #input(journal.text)
         return check_philosophy(journal.text)
     except TypeError:
         #print("Eccezione")
         return False

philosophy_files_all = []
philosophy_files_dir = []

#dir_lvl1: 000, 001, 002, ...
for num,dir_lvl1 in enumerate(os.listdir(root_dir)):
    dir_name = dir_lvl1
    print("[Dir %s: %s]"%(num, dir_name))
    dir_lvl1 = os.path.join(root_dir, dir_lvl1)
    #dir_lvl2: 0000-0001-5009-9000, 0000-0001-5018-7000, ...
    for dir_lvl2 in os.listdir(dir_lvl1):
        dir_lvl2 = os.path.join(dir_lvl1, dir_lvl2)
        #dir_lvl3: works, education, ...
        for dir_lvl3 in os.listdir(dir_lvl2):
            if dir_lvl3 == "works":
                files_dir = os.path.join(dir_lvl2, dir_lvl3)
                #file: XML file
                for file in os.listdir(files_dir):
                    file = os.path.join(files_dir, file)

                    #print(file)
                    phil = parse_XML(file)

                    if phil:
                        print("[Dir %s: %s] %s"%(num, dir_name, file))
                        philosophy_files_all.append(file)
                        philosophy_files_dir.append(file)

    if len(philosophy_files_dir) > 0:
        with open(out_dir+dir_name+".csv", "w") as f:
            for file_name in philosophy_files_dir:
                f.write(file_name+"\n")

        philosophy_files_dir = []

with open(out_file, "w") as f:
    for file in philosophy_files_all:
        f.write(file+"\n")
