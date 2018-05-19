#----------------------------------------------------------------------
# MNIST data classifier using a Conolutional Neural Network.
#
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
# from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
# from tensorflow.examples.tutorials.mnist import input_data
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# ###READ DATA###
#----------------------------------------------------------------------
#----------------------------------------------------------------------
print ("\nImporting the data")
data = np.load('example_train.npz', encoding='bytes')['data']
train_audio = [x[b'lmfcc'] for x in data]
train_l = [x[b'targets'] for x in data]
train_data = np.array(train_audio).astype(np.float32)
train_label = np.array(train_l, dtype=np.int32)
train_label = np.eye(11)[train_label.reshape(-1)]

# Load eval data
# data = np.load('extract/valid.npz', encoding='bytes')['testdata']
data = np.load('example_val.npz', encoding='bytes')['data']
val_audio = [x[b'lmfcc'] for x in data]
val_l = [x[b'targets'] for x in data]
val_data = np.array(val_audio).astype(np.float32)
val_label = np.array(val_l, dtype=np.int32)
val_label = np.eye(11)[val_label.reshape(-1)]

# Load eval data
# data = np.load('extract/valid.npz', encoding='bytes')['testdata']
data = np.load('example_test.npz', encoding='bytes')['data']
test_audio = [x[b'lmfcc'] for x in data]
test_l = [x[b'targets'] for x in data]
test_data = np.array(test_audio).astype(np.float32)
test_label = np.array(test_l, dtype=np.int32)
test_label = np.eye(11)[test_label.reshape(-1)]

# # data = np.load('../speech/newextractshuffle/trainset1.npz', encoding='bytes')['data']
# data = np.load('./newextractshuffle/trainset1.npz', encoding='bytes')['data']
# train_audio = [x[0]['lmfcc'] for x in data]
# train_l = [x[0]['targets'] for x in data]
# train_data = np.array(train_audio).astype(np.float32)
# train_label = np.array(train_l, dtype=np.int32)
# train_label = np.eye(11)[train_label.reshape(-1)]
#
# # Load eval data
# # data = np.load('../speech/extract/valid.npz', encoding='bytes')['testdata']
# data = np.load('./extract/valid.npz', encoding='bytes')['testdata']
# val_audio = [x['lmfcc'] for x in data]
# val_l = [x['targets'] for x in data]
# val_data = np.array(val_audio).astype(np.float32)
# val_label = np.array(val_l, dtype=np.int32)
# val_label = np.eye(11)[val_label.reshape(-1)]

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
batch_size = 200
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
if not os.path.exists(output_directory):
    print('\nOutput directory does not exist - creating...')
    os.makedirs(output_directory)
    os.makedirs(output_directory + '/train')
    os.makedirs(output_directory + '/val')
    print('Output directory created.')

    # initalise variables
    sess.run(tf.global_variables_initializer())
    print("initialise the model successfully")

else:

    # judge whether the model exit. If True, resore model
    if os.path.exists('mini-log/checkpoint'):
        saver.restore(sess, 'mini-log/model_at_20_epochs.ckpt-20')
        print("restore the model successfully")
    else:
        # initalise variables
        sess.run(tf.global_variables_initializer())
        print("initialise the model successfully")

    print('\nOutput directory already exists - overwriting...')
    shutil.rmtree(output_directory, ignore_errors=True)
    os.makedirs(output_directory)
    os.makedirs(output_directory + '/train')
    os.makedirs(output_directory + '/val')
    print('Output directory overwitten.')

# prepare log writers
train_writer = tf.summary.FileWriter(output_directory + '/train', sess.graph)
val_writer = tf.summary.FileWriter(output_directory + '/val')
roc_writer = tf.summary.FileWriter(output_directory)

#----------------------------------------------------------------------
# Train
#----------------------------------------------------------------------
print('\nTraining phase initiated.\n')
last_index = int(np.loadtxt('last_index'))
# print(last_index)
iteration = int(train_data.shape[0]/batch_size)
for i in range(1,training_epochs+1):
    for j in range(iteration):
        start = j*batch_size
        train_data_len = train_data.shape[0]
        batch_img, batch_lbl = nextbatch(train_data, train_label, batch_size,start,train_data_len)
        batch_img, batch_lbl = nextbatch_random(train_data, train_label, batch_size)
        val_img, val_lbl = nextbatch_random(val_data, val_label, batch_size)

        # run training step
        sess.run(train_step, feed_dict={x: batch_img,y_: batch_lbl,keep_prob: 1.0})

        # output the data into TensorBoard summaries every 10 steps
        if (j)%display_step == 0:
            train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict={x:batch_img, y_: batch_lbl,keep_prob: 1.0})
            train_writer.add_summary(train_summary, last_index+(i-1)*iteration+j)
            print("step %d, training accuracy %g"%(last_index+(i-1)*iteration+j, train_accuracy))

            val_summary, val_accuracy = sess.run([merged, accuracy], feed_dict={x: val_img,y_: val_lbl,keep_prob: 0.9})
            val_writer.add_summary(val_summary, last_index+(i-1)*iteration+j)
            print("test accuracy %g"%val_accuracy)
    # output metadata every 100 epochs
    if i % 100 == 0 or i == training_epochs:
        print('\nAdding run metadata for epoch ' + str(i) + '\n')
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        batch_img, batch_lbl = nextbatch_random(train_data, train_label, batch_size)
        summary, _ = sess.run([merged, train_step],
                              feed_dict={x:batch_img, y_: batch_lbl, keep_prob: 1.0},
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % (last_index+i*iteration))
        train_writer.add_summary(summary, last_index+i*iteration)
    # save checkpoint at the end
    if i == training_epochs:
        print('\nSaving model at ' + str(i) + ' epochs.')
        saver.save(sess, output_directory + "/model_at_" + str(i) + "_epochs.ckpt",
                   global_step=i)
np.savetxt('last_index', [last_index+training_epochs*iteration], fmt='%i')
# close writers
train_writer.close()
val_writer.close()
#----------------------------------------------------------------------
# Evaluate model
#----------------------------------------------------------------------
train_iteration = int(train_data.shape[0]/accuracy_size)
train_data_len = int(train_data.shape[0])
train_accuracy = 0
val_iteration = int(val_data.shape[0]/accuracy_size)
val_data_len = int(val_data.shape[0])
val_accuracy = 0
test_iteration = int(test_data.shape[0]/accuracy_size)
test_data_len = int(test_data.shape[0])
test_accuracy = 0
print('\nEvaluating final accuracy of the model (1/3)')
for i in range(train_iteration):
    start = i* accuracy_size
    train_d , train_l = nextbatch(train_data,train_label,accuracy_size,start,train_data_len)
    train_accuracy += sess.run(accuracy, feed_dict={x: train_d,y_: train_l,keep_prob: 1.0})
train_accuracy = train_accuracy/train_iteration

print('Evaluating final accuracy of the model (2/3)')
for i in range(val_iteration):
    start = i * accuracy_size
    val_d , val_l = nextbatch(val_data,val_label,accuracy_size,start,val_data_len)
    val_accuracy  += sess.run(accuracy, feed_dict={x: val_d, y_: val_l, keep_prob: 1.0})
val_accuracy = val_accuracy / val_iteration

print('Evaluating final accuracy of the model (3/3)')
for i in range(test_iteration):
    start = i * accuracy_size
    test_d , test_l = nextbatch(test_data,test_label,accuracy_size,start,test_data_len)
    test_accuracy += sess.run(accuracy, feed_dict={x: test_d,y_: test_l,keep_prob: 1.0})
test_accuracy = test_accuracy / test_iteration

#----------------------------------------------------------------------
# !!!UNOPTIMISED!!! ROC Curve calculation
#----------------------------------------------------------------------
'''
print('Evaluating ROC curve')
#predictions = []
labels = mnist.test.labels
threshold_num = 100
thresholds = []
fpr_list = [] #false positive rate
tpr_list = [] #true positive rate
summt = tf.Summary()
pred = tf.nn.softmax(y_conv)
predictions = sess.run(pred, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels,
                                        keep_prob: 1.0})
for i in range(len(labels)):
    threshold_step = 1. / threshold_num
    for t in range(threshold_num+1):
        th = 1 - threshold_num * t
        fp = 0
        tp = 0
        tn = 0
        for j in range(len(labels)):
            for k in range(10):
                if not labels[j][k]:
                    if predictions[j][k] >= t:
                        fp += 1
                    else:
                        tn += 1
                elif predictions[j][k].any() >= t:
                    tp += 1
        fpr = fp / float(fp + tn)
        tpr = tp / float(len(labels))
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        thresholds.append(th)

#auc = tf.metrics.auc(labels, redictions, thresholds)
summt.value.add(tag = 'ROC', simple_value = tpr)
roc_writer.add_summary(summt, fpr * 100)
roc_writer.flush()
'''
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# Output results
#----------------------------------------------------------------------
#----------------------------------------------------------------------
print ("\nTraining phase finished")
print ("\tAccuracy for the train set examples      = " , train_accuracy)
print ("\tAccuracy for the test set examples       = " , test_accuracy)
print ("\tAccuracy for the validation set examples = " , val_accuracy)

#print a statement to indicate where to find TensorBoard logs
print('\nRun "tensorboard --logdir=' + output_directory + '" to see results on localhost:6006')
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
