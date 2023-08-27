#!/usr/bin/env python3
"""contains tenserflow stuff"""
import tensorflow.compat.v1 as tf
import numpy as np


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Arguments:
    X_train -- numpy.ndarray containing the training input data
    Y_train -- numpy.ndarray containing the training labels
    X_valid -- numpy.ndarray containing the validation input data
    Y_valid -- numpy.ndarray containing the validation labels
    layer_sizes -- list containing the number of nodes in each layer of the network
    activations -- list containing the activation functions for each layer of the network
    alpha -- learning rate
    iterations -- number of iterations to train over
    save_path -- path to save the model checkpoint

    Returns:
    save_path -- the path where the model was saved
    """

    # Import necessary functions
    create_placeholders = __import__('0-create_placeholders').create_placeholders
    forward_prop = __import__('2-forward_prop').forward_prop
    calculate_loss = __import__('4-calculate_loss').calculate_loss
    calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
    create_train_op = __import__('5-create_train_op').create_train_op

    # Create placeholders
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Build the forward propagation graph
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate loss
    loss = calculate_loss(y, y_pred)

    # Create the training operation
    train_op = create_train_op(loss, alpha)

    # Calculate accuracy
    accuracy = calculate_accuracy(y, y_pred)

    # Add tensors to the graph's collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    # Initialize variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            _, train_cost, train_accuracy = sess.run([train_op, loss, accuracy], feed_dict={x: X_train, y: Y_train})

            if i % 100 == 0 or i == 0 or i == iterations:
                valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))

        # Save the model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path)

    return save_path
