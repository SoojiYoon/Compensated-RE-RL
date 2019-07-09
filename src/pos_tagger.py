import csv
from configparser import ConfigParser
from konlpy.tag import Twitter
import re
from tqdm import tqdm
tw = Twitter()

"""
output_train form
file extension: .txt
file form: e1_id / e2_id / e1 / e1 / relation / sentence
"""


config_file = "paths.ini"
conf = ConfigParser()
conf.read(config_file)
entity2id_file = conf.get("Properties","entity2id")
infile = conf.get("Preprocessing","infile")
outfile = conf.get("Preprocessing","outfile")
properties = conf.get("Properties","relation")
entity2id = {}

def tagging(s):
    e1 = s[s.find("<e1>"):s.find("</e1>")+5]
    e2 = s[s.find("<e2>"):s.find("</e2>")+5]
    s = s.replace(e1,"<(_sbj_)>",1)
    s = s.replace(e2,"<(_obj_)>",1)
    s = s.replace("[", " << ")
    s = s.replace("]", " >> ")
    tokens = tw.pos(s, norm=True, stem=True)
    s = ""
    for token in tokens:
        token = list(token)
        s += token[0] + "/" + token[1] + " "
    entities = re.findall("<</Punctuation.*?>>/Punctuation",s)
    for i in range(len(entities)):
        r=""
        t_list = entities[i].split(" ")
        for e in t_list:
            if e == "<</Punctuation" or e == ">>/Punctuation":  continue
            r += e.split("/")[0]
        r += "/Entity"
        s = s.replace(entities[i], r, 1)
    s = s.replace("<(_/Punctuation obj/Alpha _)>/Punctuation", "<<e2>>",1)
    s = s.replace("<(_/Punctuation sbj/Alpha _)>/Punctuation", "<<e1>>",1)
    e1 = e1[4:e1.find("</e1>")] + "/Entity"
    e2 = e2[4:e2.find("</e2>")] + "/Entity"
    return e1,e2,s

with open(entity2id_file,'r',encoding='utf-8') as f:
    for line in f.readlines():
        linelist = line.split()
        entity2id[linelist[0]]=linelist[1]
with open(properties,'r',encoding='utf-8') as f:
    props = f.read().strip().split("\n")
    props = set(props)
print("Number of Relations : ", len(props))
with open(infile,'r',encoding='utf-8') as f, open(outfile,'w',encoding='utf-8') as f2:
    data = list(csv.reader(f))
    for line in tqdm(data):
        rel = line[1]
        sent = line[0]
        if rel not in props:    continue
        e1,e2,tagged_s = tagging(sent)
        if len(tagged_s.split())>80:    continue
        e1_ = e1.split("/")[0]
        e2_ = e2.split("/")[0]
        if e1_ not in entity2id or e2_ not in entity2id:    continue
        e1_id = entity2id[e1_]
        e2_id = entity2id[e2_]
        # e1_id="0"
        # e2_id="0"
        try:
            int_e1id = int(e1_id)
            int_e2id = int(e2_id)
            a = tagged_s.split().index("<<e1>>")
            b = tagged_s.split().index("<<e2>>")
        except:
            continue
        write_line = "\t".join([e1_id,e2_id,e1,e2,rel,tagged_s,line[-1]])
        write_line+="\n"
        f2.write(write_line)

