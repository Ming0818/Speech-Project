import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.python.client import timeline
import numpy as np
#import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(0.0)

# mnist = input_data.read_data_sets("./MNIST_data", one_hot=True, reshape=False, validation_size=10000)


data = np.load('../speech/newextractshuffle/trainset1.npz')['data']
train_audio = [x[0]['lmfcc'] for x in data]
train_l = [x[0]['targets'] for x in data]
# data = np.load('test.npz')['testdata']
# train_audio = [x['lmfcc'] for x in data]
# train_l = [x['targets'] for x in data]
train_data = np.array(train_audio).astype(np.float32)
train_label = np.array(train_l, dtype=np.int32)
train_label = np.eye(11)[train_label.reshape(-1)]

data = np.load('../speech/extract/valid.npz')['testdata']
valid_audio = [x['lmfcc'] for x in data]
valid_l = [x['targets'] for x in data]
valid_data = np.array(valid_audio).astype(np.float32)
valid_label = np.array(valid_l, dtype=np.int32)
valid_label = np.eye(11)[valid_label.reshape(-1)]

data = np.load('../speech/extract/test.npz')['testdata']
test_audio = [x['lmfcc'] for x in data]
test_l = [x['targets'] for x in data]
test_data = np.array(test_audio).astype(np.float32)
test_label = np.array(test_l, dtype=np.int32)
test_label = np.eye(11)[test_label.reshape(-1)]

# print(mnist.train.images.shape)

def get_accuracy(pred_output, true_output):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred_output, 1), tf.argmax(true_output, 1)), tf.float32)).eval() * 100

lenet5_graph = tf.Graph()
batch_size = 64
t_label = test_label

with lenet5_graph.as_default():
    ### Training Dataset ###
    X_train_img = tf.placeholder(tf.float32, [batch_size, 399, 13, 1])
    Y_train_lbl = tf.placeholder(tf.float32, [batch_size, 11])

    ### Test Dataset ###
    train_data = np.reshape(train_data, [len(train_data), 399, 13, 1])
    test_data = np.reshape(test_data, [len(test_data), 399, 13, 1])
    X_train_img_full = tf.constant(train_data)
    X_test_img = tf.constant(test_data)

    ## Validation dataset
    # valid_data = np.reshape(valid_data, [len(valid_data), 399, 13, 1])
    # X_valid = tf.constant(valid_data)
    # Y_valid = tf.constant(valid_label)
    X_valid_img = tf.placeholder(tf.float32, [batch_size, 399, 13, 1])
    Y_valid_lbl = tf.placeholder(tf.float32, [batch_size, 11])

    ###  Hyper-parameters ###
    # learning rate
    #alpha = tf.placeholder(tf.float32)
    #alpha = tf.Variable(tf.constant(0.001, tf.float32))
    # regularization parameter
    #beta = 0.001

    ### LENET-5 Model ###
    ## Channels in layers ##
    C_conv1 = 32
    C_conv2 = 64
    N_fc1   = 1024
    N_fc2   = 11

    ## Weights and biases of layers ##
    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, C_conv1], stddev=0.1))
    B_conv1 = tf.Variable(tf.constant(0.1, tf.float32, [C_conv1]))
    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, C_conv1, C_conv2], stddev=0.1))
    B_conv2 = tf.Variable(tf.constant(0.1, tf.float32, [C_conv2]))

    W_fc1 = tf.Variable(tf.truncated_normal([98*2*64, N_fc1], stddev=0.1))
    B_fc1 = tf.Variable(tf.constant(0.1, tf.float32, [N_fc1]))
    W_fc2 = tf.Variable(tf.truncated_normal([N_fc1 , N_fc2], stddev=0.1))
    B_fc2 = tf.Variable(tf.constant(0.1, tf.float32, [N_fc2]))

    def lenet5_model(input_imgs):
        ## Layers ##
        # conv1
        conv1 = tf.nn.relu(tf.nn.conv2d(input_imgs, W_conv1, strides=[1, 1, 1, 1], padding="SAME", name="conv1") + B_conv1, name="relu1")

        # max-pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")

        # conv2
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1], padding="VALID", name="conv2") + B_conv2, name="relu2")

        # max-pool2
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")

        # fully-connected1
        fmap_shp = pool2.get_shape().as_list()
        print(fmap_shp)
        fmap_reshp = tf.reshape(pool2, [fmap_shp[0], fmap_shp[1]*fmap_shp[2]*fmap_shp[3]], name="reshape")
        fc1 = tf.nn.sigmoid(tf.matmul(fmap_reshp, W_fc1) + B_fc1, name="fc1")

        # fully-connected2 with softmax
        output = tf.matmul(fc1, W_fc2) + B_fc2

        return output, fc1

    ### Loss ###
    logits, dummy = lenet5_model(X_train_img)
    # print(logits)
    # print(dummy)
    #regularizers = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_fc3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_train_lbl)) # + beta*regularizers)

    ### Gradient Optimizer (Adagrad) ###
    grad_optimizer = tf.train.AdamOptimizer().minimize(loss)

    #wsum = tf.reduce_sum(tf.square(W_conv1)) + tf.reduce_sum(tf.square(W_conv2)) + tf.reduce_sum(tf.square(W_fc1)) + tf.reduce_sum(tf.square(W_fc2)) + tf.reduce_sum(tf.square(W_fc3))

    valid_logits, dummy = lenet5_model(X_valid_img)
    # print(valid_logits)
    # print(dummy)
    valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits, labels=Y_valid_lbl))

    ### Predictions ###
    predict_train = tf.nn.softmax(logits)
    predict_train_full = tf.nn.softmax(lenet5_model(X_train_img_full)[0])
    final_output, final_actiavtions = lenet5_model(X_test_img)
    predict_test  = tf.nn.softmax(final_output)

epochs = 4
iterations = int(np.ceil(50000/ batch_size))
# print("Train dataset: ", mnist.train.images.shape, mnist.train.labels.shape)
# print(" Validation dataset", mnist.validation.images.shape, mnist.validation.labels.shape)
# print(" Test dataset", mnist.test.images.shape, mnist.test.labels.shape)

with tf.Session(graph=lenet5_graph) as sess:
    """config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    sess = tf.Session(config=config)"""
    tf.global_variables_initializer().run()
    print("Initialization Done !!")
    cost_history = []
    steps = []
    valid_history = []
    cost_step = []
    step = 0
    for ep in range(epochs):
        for it in range(iterations):
            start = it * batch_size
            end = (it+1) * batch_size
            X_batch= np.reshape(train_data[start:end], [batch_size, 399, 13, 1])
            # print(X_batch.shape)
            Y_batch = train_label[start:end]
            # print(Y_batch.shape)
            feed = {X_train_img : X_batch, Y_train_lbl : Y_batch}
            _, cost, train_predictions = sess.run([grad_optimizer, loss, predict_train], feed_dict=feed)
            cost_history += [cost]
            # del feed
            # del X_batch
            print("Iteration: ", it, " Cost: ", cost, " Minibatch accuracy: ", get_accuracy(train_predictions, Y_batch))
            # del Y_batch
            if step%100 == 0:
                valid_cost = 0.0
                val_iterations = int(np.ceil(len(valid_data)/ batch_size))
                for val_it in range(val_iterations):
                    start = val_it * batch_size
                    end = (val_it + 1) * batch_size
                    X_batch = np.reshape(valid_data[start:end], [batch_size, 399, 13, 1])
                    Y_batch = valid_label[start:end]
                    feed = {X_valid_img: X_batch, Y_valid_lbl: Y_batch}
                    valid_cost += valid_loss.eval(session=sess, feed_dict=feed)
                print("Validation Cost:", valid_cost/val_iterations)
                valid_history += [valid_cost]
                cost_step += [cost]
                steps += [step]
            step += 1

        print("=======================================")
        test_output = predict_test.eval(session=sess)
        print("Epoch: ", ep, " Test Accuracy: ", get_accuracy(test_output, test_label))
        print("=======================================")
    test_output = predict_test.eval(session=sess)
    test_y = np.argmax(test_label, axis=1)
#    plt.plot(cost_history)
#    plt.plot(valid_history)
#    plt.show()

    pass
