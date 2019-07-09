import tensorflow as tf

class CNN:
    def __init__(self, settings, name="cnn_main"):
        self.word_dim = settings["dim_word"]
        self.pos_dim = settings["dim_pos"]
        self.num_filters = settings["num_filters"]
        self.sequence_length = settings["max_sequence_length"]
        self.num_classes = settings["num_classes"]
        self.learning_rate = settings["learning_rate"]
        self.vocab_size = settings["vocab_size"]
        self.net_name = name

    def build_model(self):
        print("\t-- build CNN model", self.net_name)
        ## Word Embedding
        self.WV = tf.Variable(tf.zeros([self.vocab_size, self.word_dim]), name="WV")

        ## Entity Position Embedding
        self.POS = tf.Variable(tf.zeros([123, self.pos_dim], ), name="POS")

        with tf.variable_scope(self.net_name):
            ##Input Placeholder
            self.input_text = tf.placeholder(tf.int32,shape=[None,self.sequence_length],name="input_text")
            self.input_y = tf.placeholder(tf.float32,shape=[None,self.num_classes],name="input_y")
            self.pos1 = tf.placeholder(tf.int32,shape=[None,self.sequence_length],name="position1")
            self.pos2 = tf.placeholder(tf.int32,shape=[None,self.sequence_length],name="position2")

            ## Word vector and Position vector embedding layer
            input_wv = tf.nn.embedding_lookup(self.WV,self.input_text)
            input_pos1 = tf.nn.embedding_lookup(self.POS,self.pos1)
            input_pos2 = tf.nn.embedding_lookup(self.POS,self.pos2)
            self.tmp_x = tf.concat(axis=2,values = [input_wv,input_pos1,input_pos2])
            x = tf.expand_dims(self.tmp_x,axis=-1)
            input_dim = x.shape.as_list()[2]

            ##Convolution & Maxpooling layer
            pool_outputs = []
            filter_size = 3
            W_f = tf.get_variable("W_f", [filter_size, input_dim, 1, self.num_filters],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_f = tf.get_variable("b_f", [self.num_filters], initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(x, W_f, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            h = tf.nn.leaky_relu(conv + b_f, name='h')
            max_len = self.sequence_length - filter_size + 1
            pool = tf.nn.max_pool(h, ksize=[1, max_len, 1, 1], strides=[1,1,1,1], padding="VALID")
            pool_outputs.append(pool)

            self.h_pools = tf.reshape(tf.concat(pool_outputs,1), [-1, self.num_filters])

            ##Fully connected layer
            W_r = tf.get_variable("W_r", [self.num_filters, self.num_classes],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_r = tf.get_variable("b_r", [self.num_classes], initializer=tf.constant_initializer(0.1))

            scores = tf.nn.xw_plus_b(self.h_pools, W_r, b_r)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=scores, labels=self.input_y))+(tf.nn.l2_loss(W_r)+tf.nn.l2_loss(b_r))*0.0001
            self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss)

            self.probabilities = tf.nn.softmax(scores)
            self.prediction = tf.argmax(self.probabilities, 1, name="prediction")
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def update(self,sents,labels,e1,e2,batch_size,sess):
        print("[START] Update RE: ",self.net_name)
        total_data_size = len(sents)
        tot_batch_size = int(total_data_size/batch_size)+1
        print("\ttotal data size: {}, total batch size: ".format(total_data_size,tot_batch_size))
        for i in range(tot_batch_size):
            st = i*batch_size
            en = min((i+1)*batch_size,total_data_size)
            batch_sent = sents[st:en]
            batch_y = labels[st:en]
            batch_pos1 = e1[st:en]
            batch_pos2 = e2[st:en]
            feed_dict = {
                self.input_text: batch_sent,
                self.input_y: batch_y,
                self.pos1:  batch_pos1,
                self.pos2:  batch_pos2
            }
            _, loss = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
        print("[END]")

