import numpy as np
from tqdm import tqdm
import utils
import re
import itertools

def load_w2v(w2v_path,w2v_dim):
    print("[LOADING] Word Vector: ",w2v_path)
    vocab_ ={}
    wv_ = []
    idx=0
    with open(w2v_path,'r',encoding='utf-8') as f:
        dim = w2v_dim
        while True:
            line = f.readline()
            if line=="":    break
            content = line.strip().split()
            word = content[0]
            if word in vocab_:  continue
            if len(content)!=dim+1: continue
            embedding = [float(i) for i in content[1:]]
            vocab_[word]=idx
            wv_.append(embedding)
            idx+=1
        vocab_["<zero>"] = idx
        wv_.append(np.zeros(dim,dtype="float32"))
    return vocab_,np.array(wv_)

def load_w2v_dict(w2v_path,w2v_dim):
    print("[LOADING] Word Vector: ",w2v_path)
    w2v={}
    with open(w2v_path,encoding='utf-8') as f:
        while True:
            line = f.readline()
            if line=="":    break
            content = line.strip().split()
            word = content[0]
            if word in w2v: continue
            if len(content)!=w2v_dim+1: continue
            embedding = [float(i) for i in content[1:]]
            w2v[word] = embedding
    return w2v

def load_bow(data_path,w2v_path,w2v_dim):
    w2v = load_w2v_dict(w2v_path,w2v_dim)
    bow={}
    emb=[]
    idx=0
    with open(data_path,encoding='utf-8') as f:
        data = f.readlines()
        for line in data:
            contents = line.strip().split("\t")
            sentence = contents[5].strip().split()
            for token in sentence:
                if token in bow:    continue
                if token not in w2v:    continue
                bow[token]=idx
                emb.append(w2v[token])
                idx+=1
    bow["<zero>"] = idx
    emb.append(np.zeros(w2v_dim,dtype="float32"))
    return bow,np.array(emb)

def const_bow(data_list,w2v_path,w2v_dim):
    w2v = load_w2v_dict(w2v_path,w2v_dim)
    bow = {}
    emb = []
    idx =0
    for line in data_list:
        for token in line:
            if token in bow:    continue
            if token not in w2v:    continue
            bow[token]=idx
            emb.append(w2v[token])
            idx+=1
    bow["<zero>"] = idx
    emb.append(np.zeros(w2v_dim,dtype="float32"))
    return bow, np.array(emb)

def load_relations(rel_path):
    print("[LOADING] Relations :", rel_path)
    with open(rel_path,'r') as f:
        rels = f.read().strip().split("\n")
    return rels

def load_entity2vec(entity_id_path, entity_vec_path):
    print("[LOADING] Entities ")
    entity2id = {}
    with open(entity_id_path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            entity2id[line[0]] = int(line[1])
    entity_vec = np.load(entity_vec_path)
    return entity2id,entity_vec

def load_relation2vec(relation_vec_path):
    print("[LOADING] Relation vector")
    relation_vec = np.load(relation_vec_path)
    return relation_vec

def load_pos_vec(pos_vec_path):
    positionVec = np.zeros((123,5),float)
    with open(pos_vec_path, 'r') as f:
        for i, r in enumerate(f.readlines()):
            r = r.strip().split("\t")
            positionVec[i] = np.array([float(x) for x in r])
    return positionVec

def pos_embed(x):
    if x< -60:  return 0
    if x>=-60 and x<=60:  return x+61
    if x>60:    return 121
def one_hot_encodding(y,num_classes):
    one_hot_y = [0.0 for _ in range(num_classes)]
    one_hot_y[y] = 1.
    return one_hot_y

def load_semeval_type_data(data_path,max_sequence_length,vocab_,labels_):
    print("[LOADING] Data: ",data_path)
    num_classes = len(labels_)
    sentences=[]
    labels=[]
    en1_pos=[]
    en2_pos=[]
    entities = []
    with open(data_path,'r',encoding='utf-8') as f:
        data = f.readlines()
        for line in tqdm(data):
            contents = line.strip().split("\t")
            sent = contents[5].split()
            if len(sent)>max_sequence_length:   continue
            e1 = contents[2].replace("/Entity","")
            e2 = contents[3].replace("/Entity","")
            rel = one_hot_encodding(labels_.index(contents[4]),num_classes)
            e1_pos,e2_pos = -1,-1
            e1_pos = sent.index("<<e1>>")
            e2_pos = sent.index("<<e2>>")
            sent[e1_pos] = e1 + "/Entity"
            sent[e2_pos] = e2 + "/Entity"

            tmp_sent=[]
            tmp_pos1=[]
            tmp_pos2=[]
            for idx,token in enumerate(sent):
                if token in vocab_:
                    tmp_sent.append(vocab_[token])
                    tmp_pos1.append(pos_embed(e1_pos-idx))
                    tmp_pos2.append(pos_embed(e2_pos-idx))
            sent_len = len(sent)
            while len(tmp_sent)!=max_sequence_length:
                tmp_sent.append(vocab_["<zero>"])
                tmp_pos1.append(122)
                tmp_pos2.append(122)
                sent_len+=1
            sentences.append(tmp_sent)
            labels.append(rel)
            en1_pos.append(tmp_pos1)
            en2_pos.append(tmp_pos2)
            entity_pair = (e1,e2)
            entities.append(entity_pair)
    return sentences,labels, en1_pos, en2_pos, entities

def constBag(entities,y_list,entity2id_dict):
    Bag = {}
    for i in range(len(y_list)):
        rel = y_list[i].index(1.)
        e1 = entities[i][0]
        e2 = entities[i][1]
        e1_id = entity2id_dict[e1]
        e2_id = entity2id_dict[e2]
        key_ = "{}(:){}(:){}".format(e1_id,e2_id,rel)
        if key_ in Bag:
            Bag[key_].append(i)
        else:
            Bag[key_]=[i]
    return Bag


def load_agent_test_data(data_path,sequence_length,num_classes,vocab_,labels,entity2_dict):
    print("[LOADING] data: ", data_path)
    data = utils.readcsv(data_path)
    sentences = []
    en1_pos = []
    en2_pos = []
    y_list = []
    entities=[]
    for line in tqdm(data):
        sent = line[0]
        entity1 = re.findall("<e1>.*?</e1>",sent)[0]
        entity2 = re.findall("<e2>.*?</e2>",sent)[0]
        entity1 = entity1.replace("<e1>","").replace("</e1>","")
        entity2 = entity2.replace("<e2>","").replace("</e2>","")
        one_hot_label = one_hot_encodding(labels.index(line[1]), num_classes)
        pos_tagged_sent, e1, e2 = utils.pos_tag(sent)
        tmp_sent = []
        tmp_pos1 = []
        tmp_pos2 = []
        for idx, token in enumerate(pos_tagged_sent):
            if token in vocab_:
                tmp_sent.append(vocab_[token])
                tmp_pos1.append(pos_embed(e1 - idx))
                tmp_pos2.append(pos_embed(e2 - idx))
        sent_len = len(pos_tagged_sent)
        while len(tmp_sent) != sequence_length:
            tmp_sent.append(vocab_["<zero>"])
            tmp_pos1.append(122)
            tmp_pos2.append(122)
            sent_len += 1
        sentences.append(tmp_sent)
        en1_pos.append(tmp_pos1)
        en2_pos.append(tmp_pos2)
        y_list.append(one_hot_label)
        entity_pair = (entity1,entity2)
        entities.append(entity_pair)
    Bags = constBag(entities,y_list,entity2_dict)
    return sentences, en1_pos, en2_pos, y_list, Bags

def load_com_train_data(data_path,max_sequence_length,vocab_,properties):
    """
    output_form : dict
    dict={rel: {sentences:[], en1_pos:[], en2_pos:[], labels:[]}, ...}
    """
    print("[LOADING] Data: ", data_path)
    result_data={}
    for rel in properties:
        sentences = []
        labels = []
        en1_pos = []
        en2_pos = []
        file_name = data_path+rel+".txt"
        with open(file_name,'r',encoding='utf-8') as f:
            for line in f.readlines():
                contents = line.strip().split("\t")
                sent = contents[5].split()
                try:
                    e1_pos = sent.index("<<e1>>")
                    e2_pos = sent.index("<<e2>>")
                except: continue
                e1 = contents[2]
                e2 = contents[3]
                sent[e1_pos] = e1
                sent[e2_pos] = e2
                if contents[-1]=="T":
                    y = [1.]
                else:
                    y = [0.]
                tmp_sent = []
                tmp_pos1 = []
                tmp_pos2 = []
                for idx, token in enumerate(sent):
                    if token in vocab_:
                        tmp_sent.append(vocab_[token])
                        tmp_pos1.append(pos_embed(e1_pos - idx))
                        tmp_pos2.append(pos_embed(e2_pos - idx))
                sent_len = len(sent)
                while len(tmp_sent) != max_sequence_length:
                    tmp_sent.append(vocab_["<zero>"])
                    tmp_pos1.append(122)
                    tmp_pos2.append(122)
                    sent_len += 1
                sentences.append(tmp_sent)
                labels.append(y)
                en1_pos.append(tmp_pos1)
                en2_pos.append(tmp_pos2)
        result_data[rel]={}
        result_data[rel]["sentences"] = sentences
        result_data[rel]["e1_pos"] = en1_pos
        result_data[rel]["e2_pos"] = en2_pos
        result_data[rel]["labels"] = labels
    return result_data


def load_extract_data(data_path,w2v_path,w2v_dim,sequence_length):
    def exEntities(list_s):
        start_positions = []
        for i,token in enumerate(list_s):
            if token.find("/Entity")!=-1:
                start_positions.append(i)
        return start_positions
    print("[LOADING] Extract data: ", data_path)
    data = utils.readfile(data_path)
    dict_origin_data={}
    dict_tagged_data={}
    tagged_sents=[]

    for line in data:
        sent = line[0]
        lbox_ids = ":".join(line[1:])
        tagged_s = utils.pos_tag_no_e1e2(sent)
        tagged_sents.append(tagged_s)
        dict_origin_data[lbox_ids] = sent
        dict_tagged_data[lbox_ids] = tagged_s
    bow,embed = const_bow(tagged_sents,w2v_path,w2v_dim)
    print("[COMPLETE] Constructing BOW")
    o_sents = []
    o_pos1 = []
    o_pos2 = []
    o_e1 = []
    o_e2 = []
    o_ids = []
    for key_, tagged_s in dict_tagged_data.items():
        if len(tagged_s)>sequence_length :  continue
        entities = exEntities(tagged_s)
        for e1_pos, e2_pos in itertools.permutations(entities,2):
            e1 = tagged_s[e1_pos].split("/En")[0]
            e2 = tagged_s[e2_pos].split("/En")[0]
            tmp_s=[]
            tmp_pos1=[]
            tmp_pos2=[]
            for idx,token in enumerate(tagged_s):
                if token in bow:
                    tmp_s.append(bow[token])
                    tmp_pos1.append(pos_embed(e1_pos-idx))
                    tmp_pos2.append(pos_embed(e2_pos-idx))
            sent_len = len(tagged_s)
            while len(tmp_s)!=sequence_length:
                tmp_s.append(bow["<zero>"])
                tmp_pos1.append(122)
                tmp_pos2.append(122)
                sent_len+=1
            o_sents.append(tmp_s)
            o_pos1.append(tmp_pos1)
            o_pos2.append(tmp_pos2)
            o_e1.append(e1)
            o_e2.append(e2)
            o_ids.append(key_)
    return bow,embed,o_sents,o_pos1,o_pos2,o_e1,o_e2,o_ids,dict_origin_data

def load_eval_data(data_path,w2v_path,w2v_dim,sequence_length,num_classes,labels):
    print("[LOADING] data: ",data_path)
    data = utils.readcsv(data_path)
    tagged_sents = []
    e1_list = []
    e2_list = []
    y_list = []
    for line in data:
        tagged_s,e1,e2 = utils.pos_tag(line[0])
        tagged_sents.append(tagged_s)
        e1_list.append(e1)
        e2_list.append(e2)
        one_hot_label = one_hot_encodding(labels.index(line[1]),num_classes)
        y_list.append(one_hot_label)

    bow, embed = const_bow(tagged_sents,w2v_path,w2v_dim)
    sentences = []
    en1_pos=[]
    en2_pos=[]

    for i in range(len(tagged_sents)):
        pos_tagged_sent = tagged_sents[i]
        e1 = e1_list[i]
        e2 = e2_list[i]

        tmp_sent = []
        tmp_pos1 = []
        tmp_pos2 = []

        for idx, token in enumerate(pos_tagged_sent):
            if token in bow:
                tmp_sent.append(bow[token])
                tmp_pos1.append(pos_embed(e1 - idx))
                tmp_pos2.append(pos_embed(e2 - idx))
        sent_len = len(pos_tagged_sent)
        while len(tmp_sent) != sequence_length:
            tmp_sent.append(bow["<zero>"])
            tmp_pos1.append(122)
            tmp_pos2.append(122)
            sent_len += 1
        sentences.append(tmp_sent)
        en1_pos.append(tmp_pos1)
        en2_pos.append(tmp_pos2)
    return sentences,en1_pos,en2_pos, y_list, bow, embed



