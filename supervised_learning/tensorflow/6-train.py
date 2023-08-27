#!/usr/bin/env python3
"""contains tenserflow stuff"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Creates placeholders for input data and labels.

    Arguments:
    nx -- number of feature columns in the data
    classes -- number of classes in the classifier

    Returns:
    x -- placeholder for input data
    y -- placeholder for one-hot labels
    """

    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")

    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")

    return x, y


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Arguments:
    y -- placeholder for the labels of the input data
    y_pred -- tensor containing the network's predictions

    Returns:
    accuracy -- tensor containing the decimal accuracy of the prediction
    """

    # Calculate the number of correct predictions
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))

    # Calculate the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy


def create_layer(prev, n, activation):
    """
    Creates a fully connected layer with specified activation.

    Arguments:
    prev -- tensor output of the previous layer
    n -- number of nodes in the layer to create
    activation -- activation function to use for the layer

    Returns:
    layer -- tensor output of the layer
    """

    # He initialization for the layer weights
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Create the layer with specified number of nodes and activation
    layer = tf.layers.dense(prev, units=n, activation=activation,
                            kernel_initializer=initializer,
                            name="layer")

    return layer


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Arguments:
    y -- placeholder for the labels of the input data
    y_pred -- tensor containing the network's predictions

    Returns:
    accuracy -- tensor containing the decimal accuracy of the prediction
    """

    # Calculate the number of correct predictions
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))

    # Calculate the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network.

    Arguments:
    loss -- the loss of the network's prediction
    alpha -- the learning rate

    Returns:
    train_op -- an operation that trains the network using gradient descent
    """

    # Create the optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)

    # Create the training operation
    train_op = optimizer.minimize(loss)

    return train_op


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.

    Arguments:
    y -- placeholder for the labels of the input data
    y_pred -- tensor containing the network's predictions

    Returns:
    loss -- tensor containing the loss of the prediction
    """

    # Calculate the softmax cross-entropy loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                          (labels=y, logits=y_pred))

    return loss


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Arguments:
    X_train -- TensorFlow placeholder for the training input data
    Y_train -- TensorFlow placeholder for the training labels
    X_valid -- TensorFlow placeholder for the validation input data
    Y_valid -- TensorFlow placeholder for the validation labels
    layer_sizes -- lists the number of nodes in each layer of the network
    activations -- lists the activation functions for each layer of the network
    alpha -- learning rate
    iterations -- number of iterations to train over
    save_path -- path to save the model checkpoint

    Returns:
    save_path -- the path where the model was saved
    """

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
            _, train_cost, train_accuracy = \
                sess.run([train_op, loss, accuracy],
                         feed_dict={x: X_train, y: Y_train})

            if i % 100 == 0 or i == 0 or i == iterations:
                valid_cost, valid_accuracy = \
                    sess.run([loss, accuracy],
                             feed_dict={x: X_valid, y: Y_valid})
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))

        # Save the model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path)

    return save_path
