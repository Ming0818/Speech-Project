from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from python_speech_features import mfcc
# number 1 to 10 data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def nsynth_generator(tfrecords_filename):
    for serialized_example in tf.python_io.tf_record_iterator(tfrecords_filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        f = example.features.feature

        audio = np.array(f['audio'].float_list.value)

        data = {
            'samplerate':
                f['sample_rate'].int64_list.value[0],
            'instrument_family':
                f['instrument_family'].int64_list.value[0],
        }

        yield data, audio


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 399, 13])   # 399x13
ys = tf.placeholder(tf.float32, [None, 11])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 399, 13, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, 32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 399x13x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 199x6x32

## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 199x6x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 99x3x64

## fc1 layer ##
W_fc1 = weight_variable([100*4*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 100*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 11])
b_fc2 = bias_variable([11])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

# Load training data
# data = np.load('test.npz')['testdata']
# train_audio = [x['lmfcc'] for x in data]
# train_l = [x['targets'] for x in data]
# train_data = np.array(train_audio).astype(np.float32)
# # train_label = np.asarray(train_l, dtype=np.int32)
# train_label = np.array(train_l, dtype=np.int32)
# train_label = np.eye(11)[train_label.reshape(-1)]

tfrecords_filename = 'nsynth-test.tfrecord'
gen_samples = nsynth_generator(tfrecords_filename)

batch_size = 100
for i in range(1000):
    train_data = []
    train_label = []
    for i in range(batch_size):
        metadata, audio = gen_samples.__next__()
        lmfcc = mfcc(audio, samplerate=metadata['samplerate'])
        train_data.append(lmfcc)
        train_label.append(metadata['instrument_family'])
    batch_xs = train_data
    batch_ys = np.eye(11)[np.array(train_label).reshape(-1)]
    print(batch_ys)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 5 == 0:
        print(compute_accuracy(
            train_data[:1000], train_label[:1000]))