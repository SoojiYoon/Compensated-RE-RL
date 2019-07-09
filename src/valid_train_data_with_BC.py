import train_compensator
import read_file
import read_file as fo
import numpy as np
import tensorflow as tf
import utils
from tqdm import tqdm
from configparser import ConfigParser

conf_path = ConfigParser()
conf_path.read("paths.ini")
conf_para = ConfigParser()
conf_para.read("parameters.ini")

train_data = conf_path.get("Preprocessing","valid_infile")
out_data = conf_path.get("Preprocessing","valid_outfile")
w2v_path = conf_path.get("Properties","w2v")
relation_path = conf_path.get("Properties","relation")
position_vec_path = conf_path.get("Properties","position_vec")
save_path = conf_path.get("Properties","relational_model")

dim_word = int(conf_para.get("Global","word_dim"))
max_sequence_length = int(conf_para.get("Global","pos_dim"))
dim_pos = int(conf_para.get("Global","pos_dim"))
batch_size = 64

vocab_dict, wv_npy = fo.load_w2v(w2v_path, dim_word)
vocab_size = len(vocab_dict.keys())
properties_list = fo.load_relations(relation_path)
position_vec_npy = fo.load_pos_vec(position_vec_path)

sents_list, y_list, en1_position_list, en2_position_list,entities_lists= fo.load_semeval_type_data(train_data,
                                                                                    80,
                                                                                    vocab_dict,properties_list)
rel_data={}
for i in range(len(properties_list)):
    rel = properties_list[i]
    rel_data[rel]=[]
    for idx,y in enumerate(y_list):
        if y[i]==1.:
            rel_data[rel].append(idx)

def convert_vec(sents,pos1,pos2):
    wv_sentences=[]
    for i in range(len(sents)):
        tmp_s = np.zeros((80,110),dtype=float)
        for j in range(80):
            word_vec = wv_npy[sents[i][j]]
            pE1 = position_vec_npy[pos1[i][j]]
            pE2 = position_vec_npy[pos2[i][j]]
            tmp = np.concatenate((word_vec,pE1,pE2),axis=-1)
            tmp_s[j] = tmp
        wv_sentences.append(tmp_s)
    return wv_sentences
valid_y = [0 for i in range(len(sents_list))]

with tf.Session() as sess:
    for rel in properties_list:
        model_name = save_path+rel
        print(model_name)
        BC = train_compensator.Binary_Classifier(vocab_size,100,5,80,0.05,rel)
        BC.build_model()
        sess.run(tf.global_variables_initializer())
        utils.load_model(sess,model_name,rel)
        rel_data_id = rel_data[rel]

        total_batch = int(len(rel_data_id)/batch_size)+1
        for b in tqdm(range(total_batch)):
            sents = []
            e1_pos = []
            e2_pos = []
            st = b*batch_size
            en = min(len(rel_data_id),(b+1)*batch_size)
            ids = rel_data_id[st:en]
            if not ids: continue
            for id in ids:
                sents.append(sents_list[id])
                e1_pos.append(en1_position_list[id])
                e2_pos.append(en2_position_list[id])
            input_data = convert_vec(sents,e1_pos,e2_pos)
            feed_dict={
                BC.input_x: input_data
            }
            y_ = sess.run(BC.predicted,feed_dict=feed_dict)
            j=0
            for id in ids:
                valid_y[id]=y_[j][0]
                j+=1
        del(BC)

    with open(train_data,'r',encoding='utf-8') as f1, open(out_data,'w',encoding='utf-8') as f2:
        for id,line in enumerate(f1.readlines()):
            content = line.strip().split("\t")
            if valid_y[id]==0:
                content.append("F")
            else:
                content.append("T")
            str_content="\t".join(content)
            f2.write(str_content+"\n")

