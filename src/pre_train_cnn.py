import tensorflow as tf
from datetime import datetime
import cnn_class
import read_file as fo
import utils
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from configparser import ConfigParser

c_para = ConfigParser()
c_path = ConfigParser()
c_para.read("parameters.ini")
c_path.read("paths.ini")

train_data = c_path.get("CNN","train_data")
w2v_data = c_path.get("Properties","w2v")
relation_data = c_path.get("Properties","relation")
position_vec_path = c_path.get("Properties","position_vec")
model_path = c_path.get("CNN","model_path")
batch_size = int(c_para.get("CNN","batch_size"))
num_epochs = int(c_para.get("CNN","num_epochs"))
cnn_setting = {}
cnn_setting["max_sequence_length"] = int(c_para.get("Global","max_sequence_length"))
cnn_setting["learning_rate"] = float(c_para.get("CNN","learning_rate"))
cnn_setting["dim_word"] = int(c_para.get("Global","word_dim"))
cnn_setting["num_classes"] = int(c_para.get("Global","num_classes"))
cnn_setting["dim_pos"] = int(c_para.get("Global","pos_dim"))
cnn_setting["num_filters"] = int(c_para.get("CNN","num_filters"))
net_name = "cnn_main"
print("[START] Pre-train CNN model: ",datetime.now())
vocab_dict, wv_npy = fo.load_bow(train_data,w2v_data,cnn_setting["dim_word"])
print(len(vocab_dict.keys()),len(wv_npy))
vocab_size = len(vocab_dict.keys())
properties_list = fo.load_relations(relation_data)
sents_list, y_list, en1_position_list, en2_position_list,entities_lists = fo.load_semeval_type_data(train_data,
                                                                                    cnn_setting["max_sequence_length"],
                                                                                    vocab_dict,properties_list)
num_classes = len(properties_list)
positionVec_npy = fo.load_pos_vec(position_vec_path)
total_data_size = len(sents_list)

print("[DONE] Loading data ")
print("[BUILD] CNN model")
cnn_setting["vocab_size"] = vocab_size
with tf.Session() as sess:
    cnn = cnn_class.CNN(cnn_setting,net_name)
    cnn.build_model()
    sess.run(tf.global_variables_initializer())
    sess.run(cnn.WV.assign(wv_npy))
    sess.run(cnn.POS.assign(positionVec_npy))
    del(wv_npy, positionVec_npy)

    total_batch_size = int(total_data_size/batch_size)+1
    for epoch in range(num_epochs):
        avg_loss=0
        avg_acc=0
        avg_f1=0
        for batch in tqdm(range(total_batch_size)):
            st = batch*batch_size
            en = min((batch+1)*batch_size,total_data_size)

            batch_sents = sents_list[st:en]
            batch_y = y_list[st:en]
            batch_pos1 = en1_position_list[st:en]
            batch_pos2 = en2_position_list[st:en]
            batch_sents, batch_y, batch_pos1, batch_pos2 = shuffle(batch_sents, batch_y, batch_pos1, batch_pos2)
            feed_dict={
                cnn.input_text: batch_sents,
                cnn.input_y:    batch_y,
                cnn.pos1:       batch_pos1,
                cnn.pos2:       batch_pos2
            }
            _, loss, acc,pred,tmp = sess.run([cnn.optimizer, cnn.loss, cnn.accuracy,cnn.prediction,cnn.tmp_x], feed_dict=feed_dict)
            avg_loss+=loss
            avg_acc+=acc
            gold_y = utils.onehot2int(batch_y)
            avg_f1+=f1_score(gold_y,pred,average="macro",labels=np.unique(gold_y))
        avg_loss/=total_batch_size
        avg_acc/=total_batch_size
        avg_f1/=total_batch_size
        print("{}, loss: {:.6f}, acc: {:.6f}, f1: {:.6f}".format(epoch,avg_loss,avg_acc,avg_f1), datetime.now())
        utils.save_model(sess,model_path,net_name)
