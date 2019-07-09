import tensorflow as tf
import numpy as np

class Agent:
    def __init__(self,policy_setting,net_name):
        self.state_dim = policy_setting["state_dim"]
        self.learning_rate = policy_setting["learning_rate"]
        self.net_name = net_name

    def build_model(self):
        print("\t-- build policy model", self.net_name)
        with tf.variable_scope(self.net_name):
            ##input placeholder
            self.state = tf.placeholder(tf.float32,shape=[None,self.state_dim], name="state")
            self.V = tf.placeholder(tf.float32,name="value")
            self.action = tf.placeholder(tf.float32, shape=[None], name="action_holder")

            ##Weighted Matrix and bias
            W_s = tf.get_variable("W_s", [self.state_dim,1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_s = tf.get_variable("b_s", [1], dtype=tf.float32, initializer=tf.constant_initializer(0.1))

            ##Policy Layer
            self.h = tf.sigmoid(tf.matmul(self.state,W_s)+b_s)
            self.a = tf.cast(self.h>0.5, dtype=tf.float32)
            policy = self.action*self.h + (1-self.action)*(1-self.h)
            # policy = tf.clip_by_value(policy,1e-8,policy)

            #Loss Layer
            self.loss = -tf.reduce_sum(tf.log(policy)*self.V)

            #Optimize Layer
            optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            self.train = optimizer.minimize(self.loss)

            #Gradient Layer
            self.tvars = tf.trainable_variables(scope=self.net_name)
            self.gradients = tf.gradients(self.loss,self.tvars)
            self.gradient_holders=[]
            for idx, var in enumerate(self.tvars):
                placeholder = tf.placeholder(tf.float32,name=str(idx)+"_holder")
                self.gradient_holders.append(placeholder)

            #Manually update gradients
            self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,self.tvars))

    def get_action(self,state,sess):
        n = np.random.random()
        h = sess.run(self.h,feed_dict={self.state:state})[0][0]
        if n>0 and n<h: return 1
        else:   return 0

    def decide_action(self,state,sess):
        action = sess.run(self.a,feed_dict={self.state:state})[0][0]
        return action

