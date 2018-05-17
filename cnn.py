#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
#from IPython.display import Audio
#from python_speech_features import mfcc
#import glob
#from itertools import groupby
#from collections import defaultdict

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # each audio are 399x13 pixels, and have one channel
  input_layer = tf.reshape(features["x"], [-1, 399, 13, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 399, 13, 1]
  # Output Tensor Shape: [batch_size, 399, 13, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 399, 13, 32]
  # Output Tensor Shape: [batch_size, 199, 6, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 199, 6, 32]
  # Output Tensor Shape: [batch_size, 199, 6, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 199, 6, 64]
  # Output Tensor Shape: [batch_size, 99, 3, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  # Input Tensor Shape: [batch_size, 99 , 3, 64]
  # Output Tensor Shape: [batch_size, 99, 3, 64]
  pool2_flat = tf.reshape(pool2, [-1, 99 * 3 * 64])

  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 99 * 3 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 11]
  logits = tf.layers.dense(inputs=dropout, units=11)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # Load training data
    data = np.load('../speech/extract/valid.npz')['testdata']
    train_audio = [x['lmfcc'] for x in data]
    train_l = [x['targets'] for x in data]
    train_data = np.array(train_audio).astype(np.float32)
    # train_label = np.asarray(train_l, dtype=np.int32)
    train_label = np.array(train_l, dtype=np.int32)

    # Load eval data
    data = np.load('../speech/extract/test.npz')['testdata']
    val_audio = [x['lmfcc'] for x in data]
    val_l = [x['targets'] for x in data]
    val_data = np.array(val_audio).astype(np.float32)
    # val_label = np.asarray(val_l, dtype=np.int32)
    val_label = np.array(val_l, dtype=np.int32)

    # Create the Estimator
    # save the model to model_dir
    audio_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    print(train_data.shape)
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
        y=train_label,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    audio_classifier.train(input_fn=train_input_fn, steps=20000,hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": val_data},
        y=val_label,
        num_epochs=1,
        shuffle=False)
    eval_results = audio_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
  tf.app.run()
