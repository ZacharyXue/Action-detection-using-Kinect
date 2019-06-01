import tensorflow as tf
import numpy as np
import random
from data_import import dataCreate 

def weight_variable(shape):
    initial = tf.truncated_normal(shape,seed=1,stddev=0.1) 
    return tf.Variable(initial,name='weight')
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name='bias')
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME',name='conv')
def max_pool(x,size):
    return tf.nn.max_pool(x,ksize=size,strides = [1,2,2,1],padding='SAME',name='max_pool')

label_size = 7

with tf.name_scope('input'):
        x_skeleton = tf.placeholder(tf.float32,[None,29,20,3],name='x_skeleton')
        x_motion = tf.placeholder(tf.float32,[None,29,20,3],name='x_motion')
        # x_skeleton = tf.placeholder(tf.float32,[None,29,14,3],name='x_skeleton')
        # x_motion = tf.placeholder(tf.float32,[None,29,14,3],name='x_motion')
        y_ = tf.placeholder(tf.float32,[None,label_size],name='y_prediction')
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')

#===============================Conv===================================
with tf.name_scope('skeleton_conv'):
        W_conv_skeleton1 = weight_variable([5,3,3,64])
        b_conv_skeleton1 = bias_variable([64])
        h_conv_skeleton1 = tf.nn.leaky_relu(conv2d(x_skeleton,W_conv_skeleton1) + b_conv_skeleton1)
        h_pool_skeleton1 = max_pool(h_conv_skeleton1,[1,3,3,1])

        W_conv_skeleton2 = weight_variable([5,3,64,128])
        b_conv_skeleton2 = bias_variable([128])
        h_conv_skeleton2 = tf.nn.leaky_relu(conv2d(h_pool_skeleton1,W_conv_skeleton2) + b_conv_skeleton2)
        h_pool_skeleton2 = max_pool(h_conv_skeleton2,[1,3,3,1])

with tf.name_scope('motion_conv'):
        W_conv_motion1 = weight_variable([5,3,3,64])
        b_conv_motion1 = bias_variable([64])
        h_conv_motion1 = tf.nn.leaky_relu(conv2d(x_motion,W_conv_motion1) + b_conv_motion1)
        h_pool_motion1 = max_pool(h_conv_motion1,[1,3,3,1])

        W_conv_motion2 = weight_variable([5,3,64,128])
        b_conv_motion2 = bias_variable([128])
        h_conv_motion2 = tf.nn.leaky_relu(conv2d(h_pool_motion1,W_conv_motion2) + b_conv_motion2)
        h_pool_motion2 = max_pool(h_conv_motion2,[1,3,3,1])

#=================================Concat================================ 
with tf.name_scope('concat'):
        x_concat = tf.concat([h_pool_skeleton2, h_pool_motion2],axis=1)

#============================fully connected============================

with tf.name_scope('fully_connected_layer'):
        W_fc1 = weight_variable([2*8*5*128,32])
        # W_fc1 = weight_variable([2*8*4*128,32])
        b_fc1 = bias_variable([32])
        x_concat_flat = tf.reshape(x_concat,[-1,2*8*5*128])
        # x_concat_flat = tf.reshape(x_concat,[-1,2*8*4*128])
        h_fc1 = tf.nn.leaky_relu(tf.matmul(x_concat_flat,W_fc1) + b_fc1 )
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

        # W_fc2 = weight_variable([64, 32])
        # b_fc2 = bias_variable([32])
        # h_fc2 = tf.nn.leaky_relu(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
        # h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)

        W_fc3 = weight_variable([32, 16])
        b_fc3 = bias_variable([16])
        h_fc3 = tf.nn.leaky_relu(tf.matmul(h_fc1_drop,W_fc3) + b_fc3)
        h_fc3_drop = tf.nn.dropout(h_fc3,keep_prob)

        W_fc4 = weight_variable([16, 8])
        b_fc4 = bias_variable([8])
        h_fc4 = tf.nn.leaky_relu(tf.matmul(h_fc3_drop,W_fc4) + b_fc4)
        h_fc4_drop = tf.nn.dropout(h_fc4,keep_prob)

        W_fc5 = weight_variable([8,label_size])
        b_fc5 = bias_variable([label_size])
        y_conv = tf.nn.leaky_relu(tf.matmul(h_fc4_drop,W_fc5) + b_fc5)

with tf.name_scope('cross_entroy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
        tf.summary.scalar('loss',cross_entropy)

with tf.name_scope('train'):
        current_iter = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.01,current_iter,decay_steps=1000,decay_rate=0.003)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy ,global_step=current_iter)

with tf.name_scope('accuracy'):
        correct_prediciton = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediciton,tf.float32))
        tf.summary.scalar('accuracy',accuracy)



merged_summary_op = tf.summary.merge_all()

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

data = dataCreate()
data.data_import()

saver = tf.train.Saver()

_batch_size = 200
with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter("CNN_Train/logs/train",sess.graph)
    test_writer = tf.summary.FileWriter("CNN_Train/logs/test",sess.graph)
    # training
    for i in range(450):
        
        batch_skeleton,batch_motion, batch_y = data.next_batch(i,_batch_size)
        test_skeleton,test_motion,test_label = data.next_batch(i,flag=1)

        sess.run(train_step, feed_dict={x_skeleton:batch_skeleton,x_motion:batch_motion, y_:batch_y, keep_prob:0.5})
        
        train_result = sess.run(merged_summary_op, feed_dict={x_skeleton:batch_skeleton,x_motion:batch_motion, y_:batch_y, keep_prob:1})
        test_result = sess.run(merged_summary_op, feed_dict={x_skeleton:test_skeleton,x_motion:test_motion, y_:test_label, keep_prob:1})
        train_writer.add_summary(train_result,i+1)
        test_writer.add_summary(test_result,i+1)

    print("Optimization Finished!")
    saver.save(sess,"CNN_Train/Model3/model.ckpt")
    # prediction
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x_skeleton:test_skeleton,x_motion:test_motion, y_:test_label, keep_prob:1}))