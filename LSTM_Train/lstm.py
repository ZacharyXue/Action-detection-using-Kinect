import tensorflow as tf
import numpy as np

import os,sys 
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir)  
from mylib.data_import import dataCreate 

tf.reset_default_graph()
## learning rate test
# Hyper Parameters
learning_rate = 0.001    # 学习率
n_hiddens = [ 32,32,32,32,32 ]         # 隐层节点数
n_classes = 7          # 输出节点数（分类数目）

# data
data = dataCreate(model='LSTM')
data.data()


# tensor placeholder
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, None, 60], name='x_input')     # 输入
    # x = tf.placeholder(tf.float32, [None, None, 42], name='x_input')     # 输入
    y = tf.placeholder(tf.float32, [None, n_classes], name='y_input')               # 输出
    keep_prob = tf.placeholder(tf.float32, name='keep_prob_input')           # 保持多少不被 dropout
    batch_size = tf.placeholder(tf.int32, [], name='batch_size_input')       # 批大小

with tf.name_scope('LSTM'):
    # weights and biases
    with tf.name_scope('weights'):
        Weights = tf.Variable(tf.truncated_normal([n_hiddens[-1], n_classes],stddev=0.1), dtype=tf.float32, name='W')
        tf.summary.histogram('output_layer_weights', Weights)
    with tf.name_scope('biases'):
        biases = tf.Variable(tf.random_normal([n_classes]), name='b')
        tf.summary.histogram('output_layer_biases', biases)

    # RNN structure
    def RNN_LSTM(x, Weights, biases):
        # # RNN 输入 reshape
        # 定义 LSTM cell
        # 实现多层 LSTM
        lstm_forward_cell = [ tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(n_hidden,\
            name='basic_lstm_cell'), output_keep_prob=keep_prob) for n_hidden in n_hiddens]
        
        lstm_backward_cell = [ tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(n_hidden,\
            name='basic_lstm_cell'), output_keep_prob=keep_prob) for n_hidden in n_hiddens]
        
        with tf.name_scope('lstm_cells_layers'):
            lstm_forward = tf.nn.rnn_cell.MultiRNNCell(lstm_forward_cell, state_is_tuple=True)
            lstm_backward = tf.nn.rnn_cell.MultiRNNCell(lstm_backward_cell, state_is_tuple=True)

        outputs, _=tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_forward,\
            cell_bw=lstm_backward,inputs=x,dtype=tf.float32)
        # 输出
        return tf.nn.softmax(tf.matmul(outputs[0][:,-1,:] + outputs[1][:,-1,:], Weights) + biases)

with tf.name_scope('output_layer'):
    pred = RNN_LSTM(x, Weights, biases)
    tf.summary.histogram('outputs', pred)
# cost
with tf.name_scope('loss'):
    cost = tf.losses.softmax_cross_entropy(y,pred)
    tf.summary.scalar('loss', cost)
# optimizer
with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

saver = tf.train.Saver()

_batch_size = 200

with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter("LSTM_Train/logs/train",sess.graph)
    test_writer = tf.summary.FileWriter("LSTM_Train/logs/test",sess.graph)
    # training
    for i in range(300):
        
        batch_x, batch_y = data.next_batch(i,_batch_size)
        test,testLabel = data.next_batch(i,flag=1)

        sess.run(train_op, feed_dict={x:batch_x, y:batch_y, keep_prob:0.5, batch_size:_batch_size})
        
        train_result = sess.run(merged, feed_dict={x:batch_x, y:batch_y, keep_prob:1.0, batch_size:_batch_size})
        test_result = sess.run(merged, feed_dict={x:test, y:testLabel, keep_prob:1.0, batch_size:_batch_size})
        train_writer.add_summary(train_result,i+1)
        test_writer.add_summary(test_result,i+1)

    print("Optimization Finished!")
    saver.save(sess,"LSTM_Train/Model4/model.ckpt")
    # prediction
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x:test, y:testLabel, keep_prob:1.0, batch_size:_batch_size}))