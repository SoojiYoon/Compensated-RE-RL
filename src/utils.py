import tensorflow as tf
import json
import csv

def save_model(sess, model_path, net_name):
    vars_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=net_name)
    saver = tf.train.Saver(vars_)
    saver.save(sess, model_path)


def load_model(sess, model_path, net_name):
    vars_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=net_name)
    saver = tf.train.Saver(vars_)
    saver.restore(sess, model_path)


def copy_network(dest_scope_name, src_scope_name):
    op_holder = []
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dst_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_vars, dst_vars in zip(src_vars, dst_vars):
        op_holder.append(dst_vars.assign(src_vars))
    return op_holder


def assign_variables(sess, dest_scope_name, src_scope_name, rate):
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dst_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    for src_vars, dst_vars in zip(src_vars, dst_vars):
        sess.run(tf.assign(dst_vars, rate * src_vars + (1 - rate) * dst_vars))


def onehot2int(data):
    result = []
    for d in data:
        result.append(d.index(1.))
    return result


def filter_selected(inputs, labels, e1_pos, e2_pos, selected_sents):
    out_inputs = []
    out_labels = []
    out_e1 = []
    out_e2 = []
    for i in selected_sents:
        out_inputs.append(inputs[i])
        out_labels.append(labels[i])
        out_e1.append(e1_pos[i])
        out_e2.append(e2_pos[i])
    return out_inputs, out_labels, out_e1, out_e2

from konlpy.tag import Twitter
import re
tw = Twitter()
def pos_tag(s):
    s = s.replace("[", " << ")
    s = s.replace("]", " >> ")
    e1 = re.findall("<e1>.*?</e1>",s)[0]
    e2 = re.findall("<e2>.*?</e2>",s)[0]
    e1_tag = e1.replace("<e1>","").replace("</e1>","").replace(" ","_")
    e1_tag += "/Entity"
    e2_tag = e2.replace("<e2>","").replace("</e2>","").replace(" ","_")
    e2_tag += "/Entity"
    s = s.replace(e1," <(_sbj_)> ")
    s = s.replace(e2," <(_obj_)> ")
    tokens = tw.pos(s, norm=True, stem=True)
    tagged_s = ""
    for token in tokens:
        token = list(token)
        tagged_s += token[0] + "/" + token[1] + " "
    entities = re.findall("<</Punctuation.*?>>/Punctuation", tagged_s)
    for i in range(len(entities)):
        r = ""
        t_list = entities[i].split(" ")
        for e in t_list:
            if e == "<</Punctuation" or e == ">>/Punctuation": continue
            r += e.split("/")[0]
        r += "/Entity"
        tagged_s = tagged_s.replace(entities[i], r, 1)
    tagged_s = tagged_s.replace("<(_/Punctuation obj/Alpha _)>/Punctuation", "<<e2>>")
    tagged_s = tagged_s.replace("<(_/Punctuation sbj/Alpha _)>/Punctuation", "<<e1>>")
    list_s = tagged_s.split()
    e1_pos = list_s.index("<<e1>>")
    e2_pos = list_s.index("<<e2>>")
    list_s[e1_pos] = e1_tag
    list_s[e2_pos] = e2_tag
    return list_s,e1_pos,e2_pos

def pos_tag_no_e1e2(s):
    s = s.replace("[", " << ")
    s = s.replace("]", " >> ")
    tokens = tw.pos(s, norm=True, stem=True)
    tagged_s = ""
    for token in tokens:
        token = list(token)
        tagged_s += token[0] + "/" + token[1] + " "
    entities = re.findall("<</Punctuation.*?>>/Punctuation", tagged_s)
    for i in range(len(entities)):
        r = ""
        t_list = entities[i].split(" ")
        for e in t_list:
            if e == "<</Punctuation" or e == ">>/Punctuation": continue
            r += e.split("/")[0]
        r += "/Entity"
        tagged_s = tagged_s.replace(entities[i], r, 1)
    list_s = tagged_s.split()
    return list_s

# useful macros
def jsonload(fname):
    with open(fname, encoding="UTF8") as f:
        return json.load(f)


def jsondump(obj, fname, split=0):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(obj, f, ensure_ascii=False, indent="\t")


def readfile(fname):
    result = []
    with open(fname, encoding="UTF8") as f:
        for line in f.readlines():
            content = line.strip().split("\t")
            result.append(content)
    return result

def readcsv(fname):
    result = []
    with open(fname,encoding="UTF8") as f:
        for line in csv.reader(f):
            result.append(line)
    return result

def writecsv(fname,data):
    with open(fname,'w',encoding='utf-8',newline='') as f:
        fw = csv.writer(f)
        for d in data:  fw.writerow(d)

def writefile(iterable, fname, processor=lambda x: x):
    with open(fname, "w", encoding="UTF8") as f:
        for item in iterable:
            f.write(processor(item) + "\n")

def distdata(infile,outfile):
    import csv
    rel_data={}
    with open(infile, 'r', encoding='utf-8') as f:
        for line in csv.reader(f):
            if line[1] in rel_data:
                rel_data[line[1]] += 1
            else:
                rel_data[line[1]] = 1

    with open(outfile, 'w', encoding='utf-8', newline='') as f:
        fw = csv.writer(f)
        for rel, v in rel_data.items():
            fw.writerow([rel, v])