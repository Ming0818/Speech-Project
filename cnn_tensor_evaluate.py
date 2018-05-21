#----------------------------------------------------------------------
# MNIST data classifier using a Conolutional Neural Network.
#
# Author: Krzysztof Furman; 15.08.2017
# TensorBoard support:
#   :scalars:
#     - accuracy
#     - wieghts and biases
#     - cost/cross entropy
#     - dropout
#   :images:
#     - reshaped input
#     - conv layers outputs
#     - conv layers weights visualisation
#   :graph:
#     - full graph of the network
#   :distributions and histograms:
#     - weights and biases
#     - activations
#   :checkpoint saving:
#     - checkpoints/saving model
#     - weights embeddings
#
#   :to be implemented:
#     - image embeddings (as in https://www.tensorflow.org/get_started/embedding_viz)
#     - ROC curve calculation (as in http://blog.csdn.net/mao_feng/article/details/54731098)
#----------------------------------------------------------------------
import tensorflow as tf
import os
import shutil
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#----------------------------------------------------------------------
#----------------------------------------------------------------------
# ###READ DATA###
#----------------------------------------------------------------------
#----------------------------------------------------------------------
print ("\nImporting the data")

# data = np.load('extract/valid.npz', encoding='bytes')['testdata']
data = np.load('example_test.npz', encoding='bytes')['data']
test_audio = [x[b'lmfcc'] for x in data]
test_l = [x[b'targets'] for x in data]
test_data = np.array(test_audio).astype(np.float32)
test_label_nohot = np.array(test_l, dtype=np.int32)
test_label = np.eye(11)[test_label_nohot.reshape(-1)]

# # Load eval data
# # data = np.load('../speech/extract/test.npz', encoding='bytes')['testdata']
# data = np.load('./extract/test.npz', encoding='bytes')['testdata']
# test_audio = [x['lmfcc'] for x in data]
# test_l = [x['targets'] for x in data]
# test_data = np.array(test_audio).astype(np.float32)
# test_label = np.array(test_l, dtype=np.int32)
# test_label = np.eye(11)[test_label.reshape(-1)]

def nextbatch(data,label,batch_n,start,data_len):
    end = min(start+batch_n,data_len)
    data = data[start:end].reshape((end-start,399,13))
    label = label[start:end].reshape((end-start,11))
    return data,label

def nextbatch_random(data,label,batch_n):
    # length = data.shape[0]
    # datatotal = np.zeros((0, data.shape[0],data.shape[1]))
    # labeltotal = np.zeros((0,label.shape[0],label.shape[1]))
    # for i in range(batch_n):
    #     randindex = np.random.randint(0, length)
    #     datatotal = np.vstack((datatotal, data[randindex]))
    #     labeltotal = np.vstack((labeltotal,label[randindex]))
    # return datatotal,labeltotal
    data_len = data.shape[0]
    num_batch = int((data_len - 1) / batch_n) + 1
    randindex = np.random.randint(0, num_batch)
    start_id = randindex * batch_n
    end_id = min((randindex + 1) * batch_n, data_len)
    x = data[start_id:end_id].reshape((end_id-start_id,399,13))
    y = label[start_id:end_id].reshape((end_id-start_id,11))
    return x,y

#----------------------------------------------------------------------
#----------------------------------------------------------------------
# ###SET THE CNN OPTIONS###
#----------------------------------------------------------------------
#----------------------------------------------------------------------
n_outputs= 11
image_x  = 399
image_y  = 13
display_step = 10
training_epochs = 20
image_shape = [-1, image_x, image_y, 1]
accuracy_size= 400
batch_size = 400
learning_rate = 1e-4
output_directory = 'mini-log/'
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# ###BUILD THE CNN###
#----------------------------------------------------------------------
#----------------------------------------------------------------------
print('\nBuilding the CNN.')
# set placeholders
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, image_x,image_y], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, n_outputs], name='y-input')
#----------------------------------------------------------------------
with tf.name_scope('input_reshape'):
    x_reshaped = tf.reshape(x, image_shape)
    tf.summary.image('input', x_reshaped, n_outputs)
#----------------------------------------------------------------------
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
#----------------------------------------------------------------------
# First conv+pool layer
#----------------------------------------------------------------------
with tf.name_scope('conv1'):
    with tf.name_scope('weights'):
        W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(W_conv1)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(W_conv1 - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(W_conv1))
            tf.summary.scalar('min', tf.reduce_min(W_conv1))
            tf.summary.histogram('histogram', W_conv1)

    with tf.name_scope('biases'):
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(b_conv1)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(b_conv1 - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(b_conv1))
            tf.summary.scalar('min', tf.reduce_min(b_conv1))
            tf.summary.histogram('histogram', b_conv1)
    with tf.name_scope('Wx_plus_b'):
        preactivated1 = tf.nn.conv2d(x_reshaped, W_conv1,strides=[1, 1, 1, 1],padding='SAME') + b_conv1
        h_conv1 = tf.nn.relu(preactivated1)
        tf.summary.histogram('pre_activations', preactivated1)
        tf.summary.histogram('activations', h_conv1)
    with tf.name_scope('max_pool'):
        h_pool1 =  tf.nn.max_pool(h_conv1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
    # save output of conv layer to TensorBoard - first 16 filters
    with tf.name_scope('Image_output_conv1'):
        image = h_conv1[0:1, :, :, 0:16]
        image = tf.transpose(image, perm=[3,1,2,0])
        tf.summary.image('Image_output_conv1', image)
    # save a visual representation of weights to TensorBoard
with tf.name_scope('Visualise_weights_conv1'):
    # We concatenate the filters into one image of row size 8 images
    W_a = W_conv1                      # i.e. [5, 5, 1, 32]
    W_b = tf.split(W_a, 32, 3)         # i.e. [32, 5, 5, 1, 1]
    rows = []
    for i in range(int(32/8)):
        x1 = i*8
        x2 = (i+1)*8
        row = tf.concat(W_b[x1:x2],0)
        rows.append(row)
    W_c = tf.concat(rows, 1)
    c_shape = W_c.get_shape().as_list()
    W_d = tf.reshape(W_c, [c_shape[2], c_shape[0], c_shape[1], 1])
    tf.summary.image("Visualize_kernels_conv1", W_d, 1024)
#----------------------------------------------------------------------
# Second conv+pool layer
#----------------------------------------------------------------------
with tf.name_scope('conv2'):
    with tf.name_scope('weights'):
        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(W_conv2)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(W_conv2 - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(W_conv2))
            tf.summary.scalar('min', tf.reduce_min(W_conv2))
            tf.summary.histogram('histogram', W_conv2)

    with tf.name_scope('biases'):
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(b_conv2)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(b_conv2 - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(b_conv2))
            tf.summary.scalar('min', tf.reduce_min(b_conv2))
            tf.summary.histogram('histogram', b_conv2)
    with tf.name_scope('Wx_plus_b'):
        preactivated2 = tf.nn.conv2d(h_pool1, W_conv2,strides=[1, 1, 1, 1],padding='VALID') + b_conv2
        h_conv2 = tf.nn.relu(preactivated2)
        tf.summary.histogram('pre_activations', preactivated2)
        tf.summary.histogram('activations', h_conv2)
    with tf.name_scope('max_pool'):
        h_pool2 =  tf.nn.max_pool(h_conv2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
    # save output of conv layer to TensorBoard - first 16 filters
    with tf.name_scope('Image_output_conv2'):
        image = h_conv2[0:1, :, :, 0:16]
        image = tf.transpose(image, perm=[3,1,2,0])
        tf.summary.image('Image_output_conv2', image)
    # save a visual representation of weights to TensorBoard
with tf.name_scope('Visualise_weights_conv2'):
    # We concatenate the filters into one image of row size 8 images
    W_a = W_conv2
    W_b = tf.split(W_a, 64, 3)
    rows = []
    for i in range(int(64/8)):
        x1 = i*8
        x2 = (i+1)*8
        row = tf.concat(W_b[x1:x2],0)
        rows.append(row)
    W_c = tf.concat(rows, 1)
    c_shape = W_c.get_shape().as_list()
    W_d = tf.reshape(W_c, [c_shape[2], c_shape[0], c_shape[1], 1])
    tf.summary.image("Visualize_kernels_conv2", W_d, 1024)
#----------------------------------------------------------------------
# Fully connected layer
#----------------------------------------------------------------------
with tf.name_scope('Fully_Connected'):
    W_fc1 = tf.Variable(tf.truncated_normal([98*2*64, 1024], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    # Flatten the output of the second pool layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 98*2*64], name="reshape")
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # Dropout
    # keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=1.0)
#----------------------------------------------------------------------
# Readout layer
#----------------------------------------------------------------------
with tf.name_scope('Readout_Layer'):
    W_fc2 = tf.Variable(tf.truncated_normal([1024, n_outputs], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[n_outputs]))
# CNN output
with tf.name_scope('Final_matmul'):
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#----------------------------------------------------------------------
# Cross entropy functions
#----------------------------------------------------------------------
with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)
#----------------------------------------------------------------------
# Optimiser
#----------------------------------------------------------------------
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
#----------------------------------------------------------------------
# Accuracy checks
#----------------------------------------------------------------------
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
print('CNN successfully built.')
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# ###START SESSSION AND COMMENCE TRAINING###
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# create session
sess = tf.Session()

# prepare checkpoint writer
saver = tf.train.Saver()

# Merge all the summaries and write them out to "mnist_logs"
merged = tf.summary.merge_all()
# judge whether the model exit. If True, resore model
if os.path.exists('mini-log/checkpoint'):
    saver.restore(sess, 'mini-log/model_at_20_epochs.ckpt-20')
    print("restore the model successfully")
else:
    print("No exist model in dir")

# os.makedirs(output_directory + '/evaluate')
# roc_writer = tf.summary.FileWriter(output_directory+'/evaluate')

#----------------------------------------------------------------------
# !!!UNOPTIMISED!!! ROC Curve calculation
#----------------------------------------------------------------------

print('Evaluating ROC curve')
#predictions = []
labels = test_label
threshold_num = 100
thresholds = []
fpr_list = [] #false positive rate
tpr_list = [] #true positive rate
summt = tf.Summary()
pred = tf.nn.softmax(y_conv)
predictions = sess.run(pred, feed_dict={x: test_data,
                                        y_: test_label,
                                        keep_prob: 1.0})
np.savez('predictions',data = predictions)
# predictions_result = np.argmax(predictions,axis=1)
# confusion = confusion_matrix(test_label_nohot,predictions_result)
# plt.pcolormesh(confusion)
# for i in range(len(labels)):
#     threshold_step = 1. / threshold_num
#     for t in range(threshold_num+1):
#         th = 1 - threshold_num * t
#         fp = 0
#         tp = 0
#         tn = 0
#         for j in range(len(labels)):
#             for k in range(11):
#                 if not labels[j][k]:
#                     if predictions[j][k] >= t:
#                         fp += 1
#                     else:
#                         tn += 1
#                 elif predictions[j][k].any() >= t:
#                     tp += 1
#         fpr = fp / float(fp + tn)
#         tpr = tp / float(len(labels))
#         fpr_list.append(fpr)
#         tpr_list.append(tpr)
#         thresholds.append(th)

# #auc = tf.metrics.auc(labels, redictions, thresholds)
# summt.value.add(tag = 'ROC', simple_value = tpr)
# roc_writer.add_summary(summt, fpr * 100)
# roc_writer.flush()

#----------------------------------------------------------------------
#----------------------------------------------------------------------
# Output results
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# print ("\nTest finished")
# print ("\tAccuracy for the test set examples       = " , test_accuracy)

# #print a statement to indicate where to find TensorBoard logs
# print('\nRun "tensorboard --logdir=' + output_directory + '" to see results on localhost:6006')
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------