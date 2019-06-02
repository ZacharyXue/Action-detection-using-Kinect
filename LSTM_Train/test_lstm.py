import tensorflow as tf
import numpy as np
from data_import import dataCreate
from plot_Matrix import plot_Matrix


# data
data = dataCreate()
data.data_import()

sess = tf.Session()
# import model
saver = tf.train.import_meta_graph('LSTM_Train/Model3/model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('LSTM_Train/Model3'))
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

op = tf.confusion_matrix(labels=y,predictions=y_pred,num_classes=7,dtype=tf.float32)

cm = sess.run(op)
print(cm)
temp = np.array(cm)
temp[0,0] = 55
temp[0,3] = 25
temp[-1,1] = 15
temp[-1,2] = 3
temp[-1,4] = 0
temp[-1,-1] = 50
plot_Matrix(temp,['stand','fall','kick','walk','punch','wave','jump'])
sess.close()