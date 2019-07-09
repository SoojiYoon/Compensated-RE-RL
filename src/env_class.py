import numpy as np

def Environment(sess,model,sents_list,y_list,en1_position_list,en2_position_list,entity_vec,relation_vec,flags=[False,False]):
    if not flags[0] and not flags[1]:
        env = Environment00(sess,model,sents_list,y_list,en1_position_list,en2_position_list,entity_vec)
    elif not flags[0] and flags[1]:
        env = Environment01(sess,model,sents_list,y_list,en1_position_list,en2_position_list,entity_vec,relation_vec)
    elif flags[0] and not flags[1]:
        env = Environment10(sess,model,sents_list,y_list,en1_position_list,en2_position_list,entity_vec)
    else:
        env = Environment11(sess,model,sents_list,y_list,en1_position_list,en2_position_list,entity_vec,relation_vec)
    return env

class Environment00:
    def __init__(self,sess,model,sents_list,y_list,en1_position_list,en2_position_list,entity_vec):
        self.sess = sess
        self.entity_vec = entity_vec
        self.total_avg_reward=0
        self.conf_scores = []
        self.sent_vecs = []
        total_train_size = len(sents_list)
        batch_size = 512
        total_batch = int(total_train_size/batch_size)+1
        for i in range(total_batch):
            st = i*batch_size
            en = min((i+1)*batch_size,total_train_size)
            batch_sents = sents_list[st:en]
            batch_pos1 = en1_position_list[st:en]
            batch_pos2 = en2_position_list[st:en]

            feed_dict = {
                model.input_text: batch_sents,
                model.pos1: batch_pos1,
                model.pos2: batch_pos2
            }
            batch_probs, batch_sent_vecs = self.sess.run([model.probabilities,model.h_pools], feed_dict=feed_dict)
            for probs,sent_vecs in zip(batch_probs,batch_sent_vecs):
                self.conf_scores.append(probs)
                self.sent_vecs.append(sent_vecs)

        for j in range(len(self.conf_scores)):
            s = np.matmul(self.conf_scores[j], y_list[j])
            self.total_avg_reward += s

        self.total_avg_reward /= total_train_size

    def init(self,e1_id,e2_id,rel,sent_ids):
        self.e1_vec = self.entity_vec[e1_id]
        self.e2_vec = self.entity_vec[e2_id]
        self.rel =rel
        self.num_selected_sents=0
        self.selected_sents = []
        self.sent_ids = sent_ids
        self.total_steps = len(sent_ids)
        self.reward=0
        self.vec_sum = np.zeros(230,dtype=float)
        state = np.concatenate((self.e1_vec,self.e2_vec,self.sent_vecs[sent_ids[0]],np.zeros(230,dtype=float)),axis=0)
        return state ## initial state

    def get_state(self,t,action):
        if t<self.total_steps:
            sent_id = self.sent_ids[t]
            sent_rep = self.sent_vecs[sent_id]
            if action==1:
                prev_sent_id = self.sent_ids[t-1]
                self.selected_sents.append(prev_sent_id)
                self.num_selected_sents+=1
                self.vec_sum += self.sent_vecs[prev_sent_id]
                self.reward+=np.matmul(self.conf_scores[prev_sent_id],self.rel)
            if self.num_selected_sents==0:
                avg_vec = np.zeros(230,dtype=float)
            else:
                avg_vec = self.vec_sum/self.num_selected_sents
            state = np.concatenate((self.e1_vec,self.e2_vec,sent_rep,avg_vec),axis=0)
            return state
        else :
            if action==1:
                self.num_selected_sents+=1
                self.selected_sents.append(self.sent_ids[-1])
                self.reward += np.matmul(self.conf_scores[self.sent_ids[-1]],self.rel)
            return None

    def get_reward(self):
        self.reward /=self.num_selected_sents
        return self.reward

    def get_comp(self,valid):
        acc=0
        for idx in self.sent_ids:
            if idx in self.selected_sents and valid[idx]==1:
                acc+=1
            elif idx not in self.selected_sents and valid[idx]==0:
                acc+=1
        return acc/len(self.sent_ids)

class Environment10:
    def __init__(self, sess, model, sents_list, y_list, en1_position_list, en2_position_list, entity_vec):
        self.sess = sess
        self.entity_vec = entity_vec
        self.total_avg_reward = 0
        self.conf_scores = []
        self.sent_vecs = []

        total_train_size = len(sents_list)
        batch_size = 512
        total_batch = int(total_train_size / batch_size) + 1

        for i in range(total_batch):
            st = i * batch_size
            en = min((i + 1) * batch_size, total_train_size)
            batch_sents = sents_list[st:en]
            batch_pos1 = en1_position_list[st:en]
            batch_pos2 = en2_position_list[st:en]

            feed_dict = {
                model.input_text: batch_sents,
                model.pos1: batch_pos1,
                model.pos2: batch_pos2
            }
            batch_probs, batch_sent_vecs = self.sess.run([model.probabilities, model.h_pools], feed_dict=feed_dict)
            for probs, sent_vecs in zip(batch_probs, batch_sent_vecs):
                self.conf_scores.append(probs)
                self.sent_vecs.append(sent_vecs)

        for j in range(len(self.conf_scores)):
            s = np.matmul(self.conf_scores[j], y_list[j])
            self.total_avg_reward += s

        self.total_avg_reward /= total_train_size

    def init(self, e1_id, e2_id, rel, sent_ids):
        self.e1_vec = self.entity_vec[e1_id]
        self.e2_vec = self.entity_vec[e2_id]
        self.rel = rel
        self.num_selected_sents = 0
        self.selected_sents = []
        self.sent_ids = sent_ids
        self.total_steps = len(sent_ids)
        self.reward = 0
        self.vec_sum = np.zeros(230, dtype=float)
        state = np.concatenate((self.e1_vec, self.e2_vec, self.sent_vecs[sent_ids[0]], np.zeros(230, dtype=float)),
                               axis=0)
        return state  ## initial state

    def get_state(self, t, action):
        if t < self.total_steps:
            sent_id = self.sent_ids[t]
            sent_rep = self.sent_vecs[sent_id]
            if action == 1:
                prev_sent_id = self.sent_ids[t - 1]
                self.selected_sents.append(prev_sent_id)
                self.num_selected_sents += 1
                self.vec_sum += self.sent_vecs[prev_sent_id]
                self.reward += np.matmul(self.conf_scores[prev_sent_id], self.rel)
            if self.num_selected_sents == 0:
                avg_vec = np.zeros(230, dtype=float)
            else:
                avg_vec = self.vec_sum / self.num_selected_sents
            state = np.concatenate((self.e1_vec, self.e2_vec, sent_rep, avg_vec), axis=0)
            return state
        else:
            if action == 1:
                self.num_selected_sents += 1
                self.selected_sents.append(self.sent_ids[-1])
                self.reward += np.matmul(self.conf_scores[self.sent_ids[-1]], self.rel)
            return None

    def set_valid(self,valid):
        self.valid = valid

    def get_reward(self):
        self.reward /= self.num_selected_sents
        return self.reward

    def get_comp(self,valid):
        acc=0
        for idx in self.sent_ids:
            if idx in self.selected_sents and valid[idx]==1:
                acc+=1
            elif idx not in self.selected_sents and valid[idx]==0:
                acc+=1
        return acc/len(self.sent_ids)

class Environment01:
    def __init__(self,sess,model,sents_list,y_list,en1_position_list,en2_position_list,entity_vec,relation_vec):
        self.sess = sess
        self.entity_vec = entity_vec
        self.relation_vec = relation_vec
        self.total_avg_reward=0
        self.conf_scores = []
        self.sent_vecs = []
        total_train_size = len(sents_list)
        batch_size = 512
        total_batch = int(total_train_size/batch_size)+1
        for i in range(total_batch):
            st = i*batch_size
            en = min((i+1)*batch_size,total_train_size)
            batch_sents = sents_list[st:en]
            batch_pos1 = en1_position_list[st:en]
            batch_pos2 = en2_position_list[st:en]

            feed_dict = {
                model.input_text: batch_sents,
                model.pos1: batch_pos1,
                model.pos2: batch_pos2
            }
            batch_probs, batch_sent_vecs = self.sess.run([model.probabilities,model.h_pools], feed_dict=feed_dict)
            for probs,sent_vecs in zip(batch_probs,batch_sent_vecs):
                self.conf_scores.append(probs)
                self.sent_vecs.append(sent_vecs)

        for j in range(len(self.conf_scores)):
            s = np.matmul(self.conf_scores[j], y_list[j])
            self.total_avg_reward += s

        self.total_avg_reward /= total_train_size

    def init(self,e1_id,e2_id,rel,sent_ids):
        self.e1_vec = self.entity_vec[e1_id]
        self.e2_vec = self.entity_vec[e2_id]
        self.one_hot_rel = rel
        self.rel_vec = self.relation_vec[np.argmax(rel)]
        self.num_selected_sents=0
        self.selected_sents = []
        self.sent_ids = sent_ids
        self.total_steps = len(sent_ids)
        self.reward=0
        self.vec_sum = np.zeros(230,dtype=float)
        state = np.concatenate((self.e1_vec,self.e2_vec,self.rel_vec,self.sent_vecs[sent_ids[0]],np.zeros(230,dtype=float)),axis=0)
        return state ## initial state

    def get_state(self,t,action):
        if t<self.total_steps:
            sent_id = self.sent_ids[t]
            sent_rep = self.sent_vecs[sent_id]
            if action==1:
                prev_sent_id = self.sent_ids[t-1]
                self.selected_sents.append(prev_sent_id)
                self.num_selected_sents+=1
                self.vec_sum += self.sent_vecs[prev_sent_id]
                self.reward+=np.matmul(self.conf_scores[prev_sent_id],self.one_hot_rel)
            if self.num_selected_sents==0:
                avg_vec = np.zeros(230,dtype=float)
            else:
                avg_vec = self.vec_sum/self.num_selected_sents
            state = np.concatenate((self.e1_vec,self.e2_vec,self.rel_vec,sent_rep,avg_vec),axis=0)
            return state
        else :
            if action==1:
                self.num_selected_sents+=1
                self.selected_sents.append(self.sent_ids[-1])
                self.reward += np.matmul(self.conf_scores[self.sent_ids[-1]],self.one_hot_rel)
            return None

    def get_reward(self):
        self.reward /=self.num_selected_sents
        return self.reward

    def get_comp(self,valid):
        acc=0
        for idx in self.sent_ids:
            if idx in self.selected_sents and valid[idx]==1:
                acc+=1
            elif idx not in self.selected_sents and valid[idx]==0:
                acc+=1
        return acc/len(self.sent_ids)

class Environment11:
    def __init__(self,sess,model,sents_list,y_list,en1_position_list,en2_position_list,entity_vec,relation_vec):
        self.sess = sess
        self.entity_vec = entity_vec
        self.relation_vec = relation_vec
        self.total_avg_reward=0
        self.conf_scores = []
        self.sent_vecs = []
        total_train_size = len(sents_list)
        batch_size = 512
        total_batch = int(total_train_size/batch_size)+1
        for i in range(total_batch):
            st = i*batch_size
            en = min((i+1)*batch_size,total_train_size)
            batch_sents = sents_list[st:en]
            batch_pos1 = en1_position_list[st:en]
            batch_pos2 = en2_position_list[st:en]

            feed_dict = {
                model.input_text: batch_sents,
                model.pos1: batch_pos1,
                model.pos2: batch_pos2
            }
            batch_probs, batch_sent_vecs = self.sess.run([model.probabilities,model.h_pools], feed_dict=feed_dict)
            for probs,sent_vecs in zip(batch_probs,batch_sent_vecs):
                self.conf_scores.append(probs)
                self.sent_vecs.append(sent_vecs)

        for j in range(len(self.conf_scores)):
            s = np.matmul(self.conf_scores[j], y_list[j])
            self.total_avg_reward += s

        self.total_avg_reward /= total_train_size

    def init(self,e1_id,e2_id,rel,sent_ids):
        self.e1_vec = self.entity_vec[e1_id]
        self.e2_vec = self.entity_vec[e2_id]
        self.one_hot_rel = rel
        self.rel_vec = self.relation_vec[np.argmax(rel)]
        self.num_selected_sents=0
        self.selected_sents = []
        self.sent_ids = sent_ids
        self.total_steps = len(sent_ids)
        self.reward=0
        self.vec_sum = np.zeros(230,dtype=float)
        state = np.concatenate((self.e1_vec,self.e2_vec,self.rel_vec,self.sent_vecs[sent_ids[0]],np.zeros(230,dtype=float)),axis=0)
        return state ## initial state

    def get_state(self,t,action):
        if t<self.total_steps:
            sent_id = self.sent_ids[t]
            sent_rep = self.sent_vecs[sent_id]
            if action==1:
                prev_sent_id = self.sent_ids[t-1]
                self.selected_sents.append(prev_sent_id)
                self.num_selected_sents+=1
                self.vec_sum += self.sent_vecs[prev_sent_id]
                self.reward+=np.matmul(self.conf_scores[prev_sent_id],self.one_hot_rel)
            if self.num_selected_sents==0:
                avg_vec = np.zeros(230,dtype=float)
            else:
                avg_vec = self.vec_sum/self.num_selected_sents
            state = np.concatenate((self.e1_vec,self.e2_vec,self.rel_vec,sent_rep,avg_vec),axis=0)
            return state
        else :
            if action==1:
                self.num_selected_sents+=1
                self.selected_sents.append(self.sent_ids[-1])
                self.reward += np.matmul(self.conf_scores[self.sent_ids[-1]],self.one_hot_rel)
            return None

    def get_reward(self):
        self.reward /=self.num_selected_sents
        return self.reward

    def get_comp(self,valid):
        acc=0
        for idx in self.sent_ids:
            if idx in self.selected_sents and valid[idx]==1:
                acc+=1
            elif idx not in self.selected_sents and valid[idx]==0:
                acc+=1
        return acc/len(self.sent_ids)
