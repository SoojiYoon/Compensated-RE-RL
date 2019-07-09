import csv
from sklearn.metrics import classification_report
import sys
import numpy as np
def exProperties(file):
    relations = []
    for line in csv.reader(file):
        relations.append(line[1])
    return relations

gold_file = "../data/ko/crowd_re_test.csv"
infile = "../result/cnn_target_addBOTH_crowd.csv"
out_report = "../report/cnn_target_addBOTH_crowd.txt"
with open(gold_file,'r',encoding='utf-8') as gold, open(infile,'r',encoding='utf-8') as predict, \
        open(out_report,'w',encoding='utf-8',newline='') as report_file:
    gold_labels = exProperties(gold)
    predic_labels = exProperties(predict)
    print("="*80)
    print(infile)
    result = classification_report(gold_labels, predic_labels,digits=4)
    # print(result)
    report_file.write(result)
    # result = result.split("\n")
    # for row in result:
    #     row = row.split()
    #     if len(row)!=5: continue
    #     row=",".join(row)
    #     report_file.write(row+"\n")

    from sklearn.metrics import precision_recall_fscore_support
    macro_p,macro_r,macro_f,_ = precision_recall_fscore_support(gold_labels,predic_labels,average='macro',labels=np.unique(gold_labels))
    micro_p,micro_r,micro_f,_ = precision_recall_fscore_support(gold_labels,predic_labels,average='micro',labels=np.unique(gold_labels))
    weighted_p,weighted_r,weighted_f,_ = precision_recall_fscore_support(gold_labels,predic_labels,average='weighted',labels=np.unique(gold_labels))
    print("{:.4f}\t{:.4f}\t{:.4f}".format(macro_p,macro_r,macro_f))
    print("{:.4f}\t{:.4f}\t{:.4f}".format(micro_p,micro_r,micro_f))
    print("{:.4f}\t{:.4f}\t{:.4f}".format(weighted_p,weighted_r,weighted_f))
    report_file.write("\tMacro: {:.4f}\t{:.4f}\t{:.4f}\n".format(macro_p,macro_r,macro_f))
    report_file.write("\tMicro: {:.4f}\t{:.4f}\t{:.4f}\n".format(micro_p,micro_r,micro_f))
    report_file.write("\tWeighted: {:.4f}\t{:.4f}\t{:.4f}\n".format(weighted_p,weighted_r,weighted_f))
