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

    # Create a placeholder for input data with shape [None, nx]
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")

    # Create a placeholder for one-hot labels with shape [None, classes]
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")

    return x, y
