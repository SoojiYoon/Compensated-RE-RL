import cnn_class
import read_file as fo
import env_class
import utils
import rl_class
import tensorflow as tf
from configparser import ConfigParser
from tqdm import tqdm
import numpy as np
import csv

conf = ConfigParser()
conf.read("pycnn_iter.ini")

test_data = conf.get("agent_test","eval_input")
result_path = conf.get("agent_test","eval_result")
w2v_data = conf.get("paths","w2v")
position_vec_path = "../data/positionVec.data"
cnn_model_path = conf.get("agent_test","cnn_model")
agent_model_path = conf.get("agent_test","agent_model")
entity_id_path = conf.get("agent_test","entity2id")
entity_vec_path = conf.get("agent_test","entity2vec")
relation_data = conf.get("agent_test","relation2id")
flag_addREL = conf.get("agent_test","flag_addREL")
flag_addCOM = conf.get("agent_test","flag_addCOM")
dim_word = int(conf.get("agent_test","dim_word"))
dim_entity = int(conf.get("agent_test","dim_entity"))
num_classes = int(conf.get("agent_test","num_classes"))

flag_addREL = True if flag_addREL=="True" else False
flag_addCOM = True if flag_addCOM=="True" else False
if flag_addREL:
    dim_relation = int(conf.get("agent_test","dim_relation"))
    relation_vec_path = conf.get("agent_test","relation2vec")
    relation_vec_npy = fo.load_relation2vec(relation_vec_path)
else:
    dim_relation = 0
    relation_vec_npy = []
vocab_dict, wv_npy = fo.load_w2v(w2v_data, dim_word)
print(len(vocab_dict.keys()), len(wv_npy))
vocab_size = len(vocab_dict.keys())

learning_rate = float(conf.get("agent_test","learning_rate"))
cnn_setting = {}
cnn_setting["max_sequence_length"] = 300
cnn_setting["learning_rate"] = 0.01
cnn_setting["dim_word"] = dim_word
cnn_setting["dim_pos"] = 5
cnn_setting["num_classes"] = num_classes
cnn_setting["num_filters"] = 230
cnn_setting["vocab_size"] = vocab_size
policy_setting = {}
policy_setting["state_dim"] = cnn_setting["num_filters"]*2 + dim_entity*2 + dim_relation
policy_setting["learning_rate"] = learning_rate


properties_list = fo.load_relations(relation_data)
positionVec_npy = fo.load_pos_vec(position_vec_path)
entity2id_dict, entity_vec_npy = fo.load_entity2vec(entity_id_path,entity_vec_path)
sents_list, en1_position_list, en2_position_list,y_list, Bags = fo.load_agent_test_data(test_data,
                                                                                        cnn_setting["max_sequence_length"],
                                                                                        num_classes,vocab_dict,
                                                                                        properties_list,
                                                                                        entity2id_dict)
selected_sents=[]
with tf.Session() as sess:
    cnn = cnn_class.CNN(cnn_setting,"cnn_target")
    agent = rl_class.Agent(policy_setting,"policy_target")
    cnn.build_model()
    agent.build_model()
    sess.run(tf.global_variables_initializer())
    utils.load_model(sess,cnn_model_path,"cnn_target")
    utils.load_model(sess,agent_model_path,"policy_target")
    sess.run(cnn.WV.assign(wv_npy))
    sess.run(cnn.POS.assign(positionVec_npy))
    env = env_class.Environment(sess,cnn,sents_list,y_list,en1_position_list,en2_position_list,entity_vec_npy,relation_vec_npy,flags=[flag_addCOM,flag_addREL])
    bag_keys = Bags.keys()
    for B in tqdm(bag_keys):
        e1_id = int(B.split("(:)")[0])
        e2_id = int(B.split("(:)")[1])
        rel = int(B.split("(:)")[2])
        sent_ids = Bags[B]
        one_hot_rel = np.zeros(num_classes,dtype=int)
        one_hot_rel[rel] = 1.0
        env.init(e1_id,e2_id,one_hot_rel,sent_ids)
        state = env.get_state(0,0)
        action = agent.decide_action([state],sess)
        for t in range(1,len(sent_ids)):
            state = env.get_state(t,action)
            action = agent.decide_action([state],sess)
        state = env.get_state(len(sent_ids),action)
        selected_sents+=env.selected_sents

selected_sents = set(selected_sents)
with open(test_data,'r',encoding='utf-8') as f1, open(result_path,'w',encoding='utf-8',newline='') as f2:
    fw = csv.writer(f2)
    for idx,line in enumerate(csv.reader(f1)):
        if idx in selected_sents:
            line[-1]="T"
        else:
            line[-1]="F"
        fw.writerow(line)