import tensorflow as tf
import numpy as np
from data_import import dataCreate


n_inputs = 60           # 输入节点数

# data
data = dataCreate(selectedLabel=[11,13,19,41],joints=n_inputs)
data.data_import()

sess = tf.Session()
# import model
saver = tf.train.import_meta_graph('LSTM_Train/Model/model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('LSTM_Train/Model'))
# initialize input
graph = tf.get_default_graph()
x_input = graph.get_tensor_by_name("inputs/x_input:0")
keep_prob = graph.get_tensor_by_name("inputs/keep_prob_input:0")

y_pred = np.array([])
y = np.array([])
for i in range(5):
    # get data
    batch_x, batch_y = data.next_batch(epoch=i,batch_size=200)
    # input data
    feed_dict = {x_input:batch_x,keep_prob:1}
    # initialize output
    output = graph.get_tensor_by_name("output_layer/Softmax:0")

    y_pred_temp = sess.run(output,feed_dict=feed_dict)

    y_pred=np.append(y_pred,np.argmax(y_pred_temp,1))
    y=np.append(y,np.argmax(batch_y,1))

op = tf.confusion_matrix(labels=y,predictions=y_pred,num_classes=4,dtype=tf.float32)
print(sess.run(op))

sess.close()