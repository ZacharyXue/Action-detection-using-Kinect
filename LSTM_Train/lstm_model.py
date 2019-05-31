import tensorflow as tf
import numpy as np
from LSTM_Train.preprocess import preprocessing as pre

class lstm():
    def __init__(self):
        self.sess = tf.Session()
        # import model
        saver = tf.train.import_meta_graph('LSTM_Train/Model2/model.ckpt.meta')
        saver.restore(self.sess,tf.train.latest_checkpoint('LSTM_Train/Model2'))
       
        graph = tf.get_default_graph()
         # initialize input
        self.x_input = graph.get_tensor_by_name("inputs/x_input:0")
        self.keep_prob = graph.get_tensor_by_name("inputs/keep_prob_input:0")
        # initialize output
        self.output = graph.get_tensor_by_name("output_layer/Softmax:0")

        self.label_dict = {0:'falling',1:'waving',2:'kicking',3:'throw'}

    def data_input(self,data):      
        data_pre = pre(pos=data).run()
        y_pred = self.sess.run(self.output,feed_dict={self.x_input:data_pre,\
            self.keep_prob:1})
        label = np.argmax(y_pred,1)
        return self.label_dict[label]
