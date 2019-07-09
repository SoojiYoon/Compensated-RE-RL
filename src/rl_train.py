import tensorflow as tf
from datetime import datetime
import cnn_class
import read_file as fo
import utils
from tqdm import tqdm
from sklearn.utils import shuffle
import rl_class
import env_class
import numpy as np
from configparser import ConfigParser

c_path = ConfigParser()
c_para = ConfigParser()
c_path.read("paths.ini")
c_para.read("parameters.ini")

train_data = c_path.get("RL","train_data")
w2v_data = c_path.get("Properties","w2v")
relation_data = c_path.get("Properties","relation")
position_vec_path = c_path.get("Properties","position_vec")
pre_cnn_path = c_path.get("RL","saved_cnn")
pre_rl_path = c_path.get("RL","saved_rl")
save_cnn_path = c_path.get("RL","target_cnn")
save_rl_path = c_path.get("RL","target_agent")
entity_id_path = c_path.get("Properties","entity2id")
entity_vec_path = c_path.get("Properties","entity2vec")

num_episodes = int(c_para.get("RL","num_episodes"))
num_samples = int(c_para.get("RL","num_samples"))
dim_word = int(c_para.get("Global","word_dim"))
dim_entity = int(c_para.get("Global","entity_dim"))
max_sequence_length = int(c_para.get("Global","max_sequence_length"))
cnn_batch_size = int(c_para.get("CNN","batch_size"))
policy_main_net_name = "policy_main"
policy_target_net_name = "policy_target"
cnn_main_net = "cnn_main"
cnn_target_net = "cnn_target"
policy_best_net_name = "policy_best"
cnn_best_net_name = "cnn_best"
flag_addREL = c_para.get("Global","add_REL")
flag_addCOM = c_para.get("Global","add_COM")
flag_addREL = True if flag_addREL=="True" else False
flag_addCOM = True if flag_addCOM=="True" else False
print("[START] Train RL model: ",datetime.now())
print("addREL : ", flag_addREL)
print("addCOM : ", flag_addCOM)

vocab_dict, wv_npy = fo.load_bow(train_data,w2v_data,dim_word)
print(len(vocab_dict.keys()),len(wv_npy))
vocab_size = len(vocab_dict.keys())
properties_list = fo.load_relations(relation_data)
sents_list, y_list, en1_position_list, en2_position_list,entities_list,a,b = fo.load_semeval_type_data(train_data,
                                                                                     max_sequence_length,
                                                                                     vocab_dict,properties_list)
entity2id_dict, entity_vec_npy = fo.load_entity2vec(entity_id_path,entity_vec_path)
if flag_addREL:
    relation_vec_path = c_path.get("Properties","relation_vec")
    relation_vec_npy = fo.load_relation2vec(relation_vec_path)
    dim_relation = int(c_para.get("Global","relation_dim"))
else:
    relation_vec_npy = []
    dim_relation = 0
positionVec_npy = fo.load_pos_vec(position_vec_path)
num_classes = len(properties_list)
Bags = fo.constBag(entities_list,y_list,entity2id_dict)
print("# of Bags: ",len(Bags))
print("[DONE] Loading data")
print("[BUILD] train RL model")
cnn_setting={}
cnn_setting["vocab_size"] = vocab_size
cnn_setting["max_sequence_length"] = max_sequence_length
cnn_setting["learning_rate"] = float(c_para.get("CNN","learning_rate"))
cnn_setting["dim_word"] =  dim_word
cnn_setting["num_classes"] = len(properties_list)
cnn_setting["dim_pos"] = int(c_para.get("Global","pos_dim"))
cnn_setting["num_filters"] = int(c_para.get("CNN","num_filters"))
policy_setting={}
policy_setting["state_dim"] = cnn_setting["num_filters"]*2 + dim_entity*2 + dim_relation
policy_setting["learning_rate"] = float(c_para.get("RL","learning_rate"))
assign_rate = float(c_para.get("RL","assign_rate"))
log_file = c_path.get("RL","log_file")
log_fo = open(log_file,'w',encoding='utf-8')
del(entity2id_dict,entities_list)
best_reward = -100000
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config = config) as sess:
    cnn_main = cnn_class.CNN(cnn_setting,cnn_main_net)
    cnn_target = cnn_class.CNN(cnn_setting,cnn_target_net)
    cnn_best = cnn_class.CNN(cnn_setting,cnn_best_net_name)
    cnn_main.build_model()
    cnn_target.build_model()
    cnn_best.build_model()
    agent_main = rl_class.Agent(policy_setting, policy_main_net_name)
    agent_target = rl_class.Agent(policy_setting, policy_target_net_name)
    agent_best = rl_class.Agent(policy_setting,policy_best_net_name)
    agent_main.build_model()
    agent_target.build_model()
    agent_best.build_model()
    sess.run(tf.global_variables_initializer())
    utils.load_model(sess,pre_cnn_path,cnn_main_net)
    utils.load_model(sess,pre_rl_path,policy_main_net_name)
    sess.run(cnn_main.WV.assign(wv_npy))
    sess.run(cnn_main.POS.assign(positionVec_npy))
    sess.run(cnn_target.WV.assign(wv_npy))
    sess.run(cnn_target.POS.assign(positionVec_npy))
    sess.run(utils.copy_network(cnn_target_net,cnn_main_net))
    sess.run(utils.copy_network(policy_target_net_name,policy_main_net_name))
    del(wv_npy,positionVec_npy)
    grad_buffer = sess.run(agent_main.tvars)
    k = list(Bags.keys())
    for episode in range(num_episodes):
        env = env_class.Environment(sess, cnn_target, sents_list, y_list, en1_position_list, en2_position_list,
                                   entity_vec_npy,relation_vec_npy,flags=[flag_addCOM,flag_addREL])
        total_avg_reward = env.total_avg_reward
        print("total_avg_reward: {:.6f}".format(total_avg_reward))
        selected_sents=[]
        avg_loss=0
        avg_reward=0
        tot_selected_sents=0
        shuffled_bag_key = shuffle(k)
        for idx,grad in enumerate(grad_buffer): grad_buffer[idx]=grad*0

        for B in tqdm(shuffled_bag_key):
            e1_id = int(B.split("(:)")[0])
            e2_id = int(B.split("(:)")[1])
            rel = int(B.split("(:)")[2])
            one_hot_rel = np.zeros(num_classes,dtype=int)
            one_hot_rel[rel] = 1
            sent_ids = Bags[B]
            sent_ids = shuffle(sent_ids)
            sampled_states=[[] for _ in range(num_samples)]
            sampled_actions=[[] for _ in range(num_samples)]
            sampled_reward=[ 0 for _ in range(num_samples)]
            sample_avg_reward=0
            for x in range(num_samples):
                env.init(e1_id, e2_id, one_hot_rel, sent_ids)
                state = env.get_state(0,0)
                sampled_states[x].append(state)
                action = agent_target.get_action([state],sess)
                sampled_actions[x].append(action)
                for t in range(1,len(sent_ids)):
                    state = env.get_state(t,action)
                    sampled_states[x].append(state)
                    action = agent_target.get_action([state],sess)
                    sampled_actions[x].append(action)
                state = env.get_state(len(sent_ids),action)
                if env.num_selected_sents==0:
                    reward = total_avg_reward
                else:
                    reward = env.get_reward()
                sample_avg_reward += reward/num_samples
                sampled_reward[x] = reward

            ##train main agent
            tmp_loss=0
            for x in range(num_samples):
                feed_dict = {
                    agent_main.state: sampled_states[x],
                    agent_main.action:  sampled_actions[x],
                    agent_main.V:   (sampled_reward[x]-sample_avg_reward)*100
                }
                # update agent automatically
                # _, loss = sess.run([agent_main.train, agent_main.loss],feed_dict=feed_dict)
                # tmp_loss+=loss

                ## calculate gradients manually
                grads, loss = sess.run([agent_main.gradients,agent_main.loss],feed_dict=feed_dict)
                for idx,grad in enumerate(grads):
                    grad_buffer[idx]+=grad
                tmp_loss+=loss
            avg_loss+=tmp_loss/num_samples

            env.init(e1_id,e2_id,one_hot_rel,sent_ids)
            state = env.get_state(0,0)
            action = agent_target.decide_action([state], sess)
            for t in range(1,len(sent_ids)):
                state = env.get_state(t, action)
                action = agent_target.decide_action([state], sess)
            state = env.get_state(len(sent_ids),action)
            if env.num_selected_sents!=0:
                tot_selected_sents+=env.num_selected_sents
                avg_reward+=env.get_reward()
                selected_sents+=env.selected_sents
            else:
                avg_reward+=total_avg_reward
        tmp_sents,tmp_y,tmp_e1,tmp_e2 = utils.filter_selected(sents_list,y_list,en1_position_list,en2_position_list,selected_sents)
        selected_sents_list, selected_y_list, selected_e1_list,selected_e2_list = shuffle(tmp_sents, tmp_y, tmp_e1, tmp_e2)
        if tot_selected_sents!=0:
            cnn_main.update(selected_sents_list,selected_y_list,selected_e1_list,selected_e2_list,cnn_batch_size,sess)
        avg_loss/=len(shuffled_bag_key)
        avg_reward/=len(shuffled_bag_key)
        log = "episode {}, avg_loss = {:.6f}, avg_reward = {:.6f}, tot_reward = {:.6f}, selected_sents = {}".format(episode,avg_loss,avg_reward,total_avg_reward, tot_selected_sents)
        print(log)
        log_fo.write(log+"\n")

        if total_avg_reward>best_reward:
            best_reward = total_avg_reward
            sess.run(utils.copy_network(policy_best_net_name,policy_target_net_name))
            sess.run(utils.copy_network(cnn_best_net_name,cnn_target_net))
        if episode%3==0:
            utils.save_model(sess,save_rl_path+"training",policy_target_net_name)
            utils.save_model(sess,save_cnn_path+"training",cnn_target_net)

        feed_dict = dict(zip(agent_main.gradient_holders,grad_buffer))
        _ = sess.run(agent_main.update_batch,feed_dict=feed_dict)
        utils.assign_variables(sess, policy_target_net_name, policy_main_net_name, assign_rate)
        utils.assign_variables(sess, cnn_target_net, cnn_main_net, assign_rate)
        del(env)

    print("[END]")
    sess.run(utils.copy_network(policy_target_net_name,policy_best_net_name))
    sess.run(utils.copy_network(cnn_target_net,cnn_best_net_name))
    utils.save_model(sess, save_rl_path, policy_target_net_name)
    utils.save_model(sess, save_cnn_path, cnn_target_net)
    log_fo.close()