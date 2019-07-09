import tensorflow as tf
from datetime import datetime
import cnn_class
import read_file as fo
import utils
from tqdm import tqdm
from sklearn.utils import shuffle
import rl_class
import numpy as np
import env_class
from configparser import ConfigParser

c_path = ConfigParser()
c_para = ConfigParser()
c_path.read("paths.ini")
c_para.read("parameters.ini")

train_data = c_path.get("Agent","train_data")
w2v_data = c_path.get("Properties","w2v")
relation_data = c_path.get("Properties","relation")
position_vec_path = c_path.get("Properties","position_vec")
cnn_model_path = c_path.get("Agent","saved_cnn")
rl_model_path = c_path.get("Agent","model_path")
entity_id_path = c_path.get("Properties","entity2id")
entity_vec_path = c_path.get("Properties","entity2vec")

num_episodes = int(c_para.get("Agent","num_episodes"))
num_samples = int(c_para.get("Agent","num_samples"))
dim_word = int(c_para.get("Global","word_dim"))
dim_entity = int(c_para.get("Global","entity_dim"))
max_sequence_length = int(c_para.get("Global","max_sequence_length"))
policy_main_net_name = "policy_main"
policy_target_net_name = "policy_target"
policy_best_net_name = "policy_best"
log_file = c_path.get("Agent","log_file")
fo_log = open(log_file,'w',encoding='utf-8')
flag_addREL = c_para.get("Global","add_REL")
flag_addCOM = c_para.get("Global","add_COM")
flag_addREL = True if flag_addREL=="True" else False
flag_addCOM = True if flag_addCOM=="True" else False
print("[START] Pre-train Policy model: , ", datetime.now())
print("addREL : ", flag_addREL)
print("addCOM : ", flag_addCOM)

vocab_dict, wv_npy = fo.load_bow(train_data,w2v_data, dim_word)
print(len(vocab_dict.keys()), len(wv_npy))
vocab_size = len(vocab_dict.keys())
properties_list = fo.load_relations(relation_data)
sents_list, y_list, en1_position_list, en2_position_list, entities_list = fo.load_semeval_type_data(train_data,
                                                                                                    max_sequence_length,
                                                                                                    vocab_dict,
                                                                                                    properties_list)
entity2id_dict, entity_vec_npy = fo.load_entity2vec(entity_id_path, entity_vec_path)
if flag_addREL:
    relation_vec_path = c_path.get("Properties","relation_vec")
    relation_vec_npy = fo.load_relation2vec(relation_vec_path)
    dim_relation = int(c_para.get("Global","relation_dim"))
else:
    relation_vec_npy = []
    dim_relation = 0
if flag_addCOM:
    valid = []
    with open(train_data, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            content = line.strip().split("\t")
            if content[-1] == "T":
                valid.append(1)
            else:
                valid.append(0)
else:
    valid=[-1 for i in range(len(sents_list))]

positionVec_npy = fo.load_pos_vec(position_vec_path)
num_classes = len(properties_list)
Bags = fo.constBag(entities_list, y_list, entity2id_dict)
print("# of Bags: ", len(Bags))
print("[DONE] Loading data")
print("[BUILD] pre-trained CNN model")
cnn_setting = {}
cnn_setting["vocab_size"] = vocab_size
cnn_setting["max_sequence_length"] = max_sequence_length
cnn_setting["learning_rate"] = float(c_para.get("CNN","learning_rate"))
cnn_setting["dim_word"] = dim_word
cnn_setting["num_classes"] = int(c_para.get("Global","num_classes"))
cnn_setting["dim_pos"] = int(c_para.get("Global","pos_dim"))
cnn_setting["num_filters"] = int(c_para.get("CNN","num_filters"))
cnn_net_name = "cnn_main"
policy_setting = {}
policy_setting["state_dim"] = cnn_setting["num_filters"]*2 + dim_entity*2 + dim_relation
policy_setting["learning_rate"] = float(c_para.get("Agent","learning_rate"))
assign_rate = float(c_para.get("Agent","assign_rate"))
del (entity2id_dict, entities_list)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config = config) as sess:
    cnn = cnn_class.CNN(cnn_setting, cnn_net_name)
    cnn.build_model()
    agent_main = rl_class.Agent(policy_setting, policy_main_net_name)
    agent_target = rl_class.Agent(policy_setting, policy_target_net_name)
    agent_best = rl_class.Agent(policy_setting,policy_best_net_name)
    agent_main.build_model()
    agent_target.build_model()
    agent_best.build_model()
    sess.run(tf.global_variables_initializer())
    utils.load_model(sess, cnn_model_path, cnn_net_name)

    sess.run(cnn.WV.assign(wv_npy))
    sess.run(cnn.POS.assign(positionVec_npy))
    sess.run(utils.copy_network(policy_target_net_name, policy_main_net_name))
    env = env_class.Environment(sess, cnn, sents_list, y_list, en1_position_list,
                                en2_position_list, entity_vec_npy,relation_vec_npy,flags=[flag_addCOM,flag_addREL])
    del (wv_npy, positionVec_npy, entity_vec_npy, sents_list, y_list, en1_position_list, en2_position_list)
    total_avg_reward = env.total_avg_reward
    k = list(Bags.keys())
    best_reward = -10000
    grad_buffer = sess.run(agent_main.tvars)
    print(total_avg_reward)
    for episode in range(num_episodes):
        avg_loss = 0
        avg_reward = 0
        tot_selected_sents = 0
        shuffled_bag_key = shuffle(k)
        for idx, grad in enumerate(grad_buffer):
            grad_buffer[idx] = grad * 0

        for B in tqdm(shuffled_bag_key):
            e1_id = int(B.split("(:)")[0])
            e2_id = int(B.split("(:)")[1])
            rel = int(B.split("(:)")[2])
            one_hot_rel = np.zeros(num_classes, dtype=int)
            one_hot_rel[rel] = 1
            sent_ids = Bags[B]
            sent_ids = shuffle(sent_ids)
            sampled_states = [[] for _ in range(num_samples)]
            sampled_actions = [[] for _ in range(num_samples)]
            sampled_reward = [0 for _ in range(num_samples)]
            sample_avg_reward = 0
            for x in range(num_samples):
                env.init(e1_id, e2_id, one_hot_rel, sent_ids)
                state = env.get_state(0, 0)
                sampled_states[x].append(state)
                action = agent_target.get_action([state], sess)
                sampled_actions[x].append(action)
                for t in range(1, len(sent_ids)):
                    state = env.get_state(t, action)
                    sampled_states[x].append(state)
                    action = agent_target.get_action([state], sess)
                    sampled_actions[x].append(action)
                state = env.get_state(len(sent_ids), action)
                if env.num_selected_sents == 0:
                    reward = total_avg_reward
                else:
                    reward = env.get_reward()
                sample_avg_reward += reward / num_samples
                sampled_reward[x] = reward

            ##train main agent
            tmp_loss = 0
            for x in range(num_samples):
                feed_dict = {
                    agent_main.state: sampled_states[x],
                    agent_main.action: sampled_actions[x],
                    agent_main.V: (sampled_reward[x] - sample_avg_reward) * 100
                }
                ## update agent automatically
                # _, loss = sess.run([agent_main.train, agent_main.loss], feed_dict=feed_dict)
                # tmp_loss += loss

                ## calculate gradients
                grads,loss = sess.run([agent_main.gradients,agent_main.loss],feed_dict=feed_dict)
                for idx,grad in enumerate(grads):
                    grad_buffer[idx]+=grad
                tmp_loss+=loss

            avg_loss += tmp_loss / num_samples

            env.init(e1_id, e2_id, one_hot_rel, sent_ids)
            state = env.get_state(0, 0)
            action = agent_target.decide_action([state], sess)
            for t in range(1, len(sent_ids)):
                state = env.get_state(t, action)
                action = agent_target.decide_action([state], sess)
            state = env.get_state(len(sent_ids), action)
            if env.num_selected_sents != 0:
                tot_selected_sents += env.num_selected_sents
                avg_reward += (env.get_reward()+env.get_comp(valid))
            else:
                avg_reward += (total_avg_reward+env.get_comp(valid))

        avg_loss /= len(shuffled_bag_key)
        avg_reward /= len(shuffled_bag_key)
        log = "episode {}, avg_loss = {:.6f}, avg_reward = {:.6f}, tot_reward = {:.6f}, selected_sents = {}".format(
            episode, avg_loss, avg_reward, total_avg_reward, tot_selected_sents)
        print(log)
        fo_log.write(log+"\n")
        if avg_reward>best_reward:
            best_reward=avg_reward
            sess.run(utils.copy_network(policy_best_net_name,policy_target_net_name))
        if episode%5==0:
            utils.save_model(sess,rl_model_path+"training",policy_target_net_name)

        feed_dict = dict(zip(agent_main.gradient_holders,grad_buffer))
        _ = sess.run(agent_main.update_batch,feed_dict=feed_dict)

        utils.assign_variables(sess, policy_target_net_name, policy_main_net_name, assign_rate)

    print("[END]")
    sess.run(utils.copy_network(policy_main_net_name,policy_best_net_name))
    utils.save_model(sess, rl_model_path, policy_main_net_name)
    fo_log.close()
