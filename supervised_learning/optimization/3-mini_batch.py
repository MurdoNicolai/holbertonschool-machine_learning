#!/usr/bin/env python3
""" contains normalization"""
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data
tf.disable_eager_execution()


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ train a mini-batch with X_train data split into Batch_size batches"""
    tf.reset_default_graph()

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
        train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
        valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
        valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

        print("After {} epochs:".format(0))
        print("\tTraining Cost: {}\n\tTraining Accuracy: {}".format(
              train_cost, train_accuracy))
        print("\tValidation Cost: {}\n\tValidation Accuracy: {}".format(
              valid_cost, valid_accuracy))

        m = X_train.shape[0]
        for epoch in range(epochs):
            X_train, Y_train = shuffle_data(X_train, Y_train)

            for step in range(0, m - 1, batch_size):
                end = step + batch_size
                X_batch = X_train[step:end]
                Y_batch = Y_train[step:end]

                if step % (100 * batch_size) == 0 and (step != 0):
                    print("\tStep {}:\n\t\tCost: {}\n\t\tAccuracy: {}".format(
                          int(step/batch_size), step_cost, step_accuracy))
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                step_cost = sess.run(loss,
                                     feed_dict={x: X_batch, y: Y_batch})
                step_accuracy = sess.run(accuracy,
                                         feed_dict={x: X_batch, y: Y_batch})

            train_cost = sess.run(loss,
                                  feed_dict={x: X_train, y: Y_train})
            train_accuracy = sess.run(accuracy,
                                      feed_dict={x: X_train, y: Y_train})
            valid_cost = sess.run(loss,
                                  feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(accuracy,
                                      feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch + 1))
            print("\tTraining Cost: {}\n\tTraining Accuracy: {}".format(
                  train_cost, train_accuracy))
            print("\tValidation Cost: {}\n\tValidation Accuracy: {}".format(
                  valid_cost, valid_accuracy))

            train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch + 1))
            print("\tTraining Cost: {}\n\tTraining Accuracy: {}".format(
                  train_cost, train_accuracy))
            print("\tValidation Cost: {}\n\tValidation Accuracy: {}".format(
                  valid_cost, valid_accuracy))
            return 0

        saver.save(sess, save_path)

    return save_path
