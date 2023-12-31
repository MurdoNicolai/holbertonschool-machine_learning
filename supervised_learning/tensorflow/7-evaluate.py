#!/usr/bin/env python3
"""contains tenserflow stuff"""
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Arguments:
    X -- TensorFlow placeholder for the input data to evaluate
    Y -- TensorFlow placeholder for the one-hot labels for X
    save_path -- location to load the model from

    Returns:
    prediction -- network's prediction
    accuracy -- accuracy of the prediction
    loss -- loss of the prediction
    """

    # Load the model
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        # Get tensors from the graph's collection
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]

        # Evaluate the model on the input data
        prediction, eva_los, eval_accuracy = sess.run([y_pred, loss, accuracy],
                                                      feed_dict={x: X, y: Y})

    return prediction, eval_accuracy, eva_los
