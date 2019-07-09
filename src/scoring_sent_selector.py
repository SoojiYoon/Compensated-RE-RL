import csv
from sklearn.metrics import classification_report,precision_recall_fscore_support,accuracy_score
import utils
import sys

def exTag(data):
    tags=[]
    for line in data:
        tags.append(line[-1])
    return tags
def rel_data(gold_data,pred_data):
    dict_data = {}
    for g_line,p_line in zip(gold_data,pred_data):
        rel = g_line[1]
        if rel in dict_data:
            dict_data[rel]["gold_data"].append(g_line[-1])
            dict_data[rel]["pred_data"].append(p_line[-1])
        else:
            dict_data[rel]={}
            dict_data[rel]["gold_data"] = [g_line[-1]]
            dict_data[rel]["pred_data"] = [p_line[-1]]
    return dict_data
goldfile = "../data/ko/agent_test.csv"
predfile = "../result/agent_target_addBOTH_44_result.csv"

gold_data = utils.readcsv(goldfile)
pred_data = utils.readcsv(predfile)

gold_tags = exTag(gold_data)
pred_tags = exTag(pred_data)
print(classification_report(gold_tags,pred_tags,digits=4))
macro_p,macro_r,macro_f1,_ = precision_recall_fscore_support(gold_tags,pred_tags,average="macro")
micro_p,micro_r,micro_f1,_ = precision_recall_fscore_support(gold_tags,pred_tags,average="micro")
acc = accuracy_score(gold_tags,pred_tags)
print("Macro: {:.4f}\t{:.4f}\t{:.4f}".format(macro_p,macro_r,macro_f1))
print("Micro: {:.4f}\t{:.4f}\t{:.4f}".format(micro_p,micro_r,micro_f1))
print("acc: {:.4f}".format(acc))

dict_data = rel_data(gold_data,pred_data)
with open("../a.txt",'w',encoding='utf-8') as f:
    for rel,dict_tags in dict_data.items():
        gold_tag = dict_tags["gold_data"]
        pred_tag = dict_tags["pred_data"]
        p,r,f1,_ = precision_recall_fscore_support(gold_tag,pred_tag,average="macro")
        acc = accuracy_score(gold_tag,pred_tag)
        out_ = "{}\t{:.4f}\t{:.4f}\t{}".format(rel,f1,acc,len(gold_tag))
        print(out_)
        f.write(out_+"\n")