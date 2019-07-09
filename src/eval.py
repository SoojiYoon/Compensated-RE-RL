import tensorflow as tf
import cnn_class
import read_file as fo
import utils
from tqdm import tqdm
from configparser import ConfigParser

conf = ConfigParser()
conf.read("pycnn_iter.ini")

test_data = conf.get("paths","eval_input")
w2v_data = conf.get("paths","w2v")

position_vec_path = "../data/positionVec.data"
model_path = conf.get("paths","model")
output_path = conf.get("paths","eval_output")
relation_data = conf.get("paths","relation")
batch_size=int(conf.get("parameters","batch_size"))
cnn_setting={}
cnn_setting["max_sequence_length"] = 300
cnn_setting["learning_rate"] = 0.01
cnn_setting["dim_word"] = 100
cnn_setting["dim_pos"] = 5
cnn_setting["num_classes"] = 44
cnn_setting["num_filters"] = 230
type = conf.get("parameters","type")

if type=="CNN":
    net_name = "cnn_main"
else:
    net_name = "cnn_target"

print("[START] Test model: ", model_path)
print("test data: ",test_data)

properties_list = fo.load_relations(relation_data)
sents_list, en1_position_list, en2_position_list, y_list, vocab, wv_npy = fo.load_eval_data(test_data,w2v_data,cnn_setting["dim_word"],cnn_setting["max_sequence_length"],cnn_setting["num_classes"],properties_list)
num_classes = len(properties_list)
positionVec_npy = fo.load_pos_vec(position_vec_path)
total_data_size = len(sents_list)
print("[DONE] Loading data")
print("[BUILD] CNN model")
vocab_size = len(vocab.keys())
cnn_setting["vocab_size"] = vocab_size

with tf.Session() as sess:
    cnn = cnn_class.CNN(cnn_setting,net_name)
    cnn.build_model()
    sess.run(tf.global_variables_initializer())
    utils.load_model(sess,model_path,net_name)
    sess.run(cnn.WV.assign(wv_npy))
    sess.run(cnn.POS.assign(positionVec_npy))

    total_batch_size = int(total_data_size/batch_size)+1
    predicted_y = []
    probs=[]
    for batch in tqdm(range(total_batch_size)):
        st = batch*batch_size
        en = min((batch+1)*batch_size,total_data_size)

        batch_sents = sents_list[st:en]
        batch_y = y_list[st:en]
        batch_pos1 = en1_position_list[st:en]
        batch_pos2 = en2_position_list[st:en]
        feed_dict = {
            cnn.input_text: batch_sents,
            cnn.input_y: batch_y,
            cnn.pos1: batch_pos1,
            cnn.pos2: batch_pos2
        }

        prediction,probabilities = sess.run([cnn.prediction,cnn.probabilities], feed_dict=feed_dict)
        for i in range(len(batch_sents)):
            predicted_y.append(properties_list[prediction[i]])
            probs.append(probabilities[i][prediction[i]])


result=[]
eval_data = utils.readcsv(test_data)
for i,line in enumerate(eval_data):
    w = [line[0],predicted_y[i],line[2],line[3],line[4],probs[i]]
    result.append(w)

utils.writecsv(output_path,result)
