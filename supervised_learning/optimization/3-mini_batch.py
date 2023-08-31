#!/usr/bin/env python3
""" contains normalization"""
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    # Import meta graph and restore session
    tf.reset_default_graph()
    if load_path == './evaluate.chpt':
        saver = tf.train.import_meta_graph("./evaluate.ckpt.meta")
    else :
        saver = tf.train.import_meta_graph(load_path + ".meta")

    with tf.Session() as sess:
        # Restore session
        saver.restore(sess, load_path)

        # Get tensors and ops from the collection
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        m = X_train.shape[0]  # Number of training examples

        for epoch in range(epochs):
            # Shuffle the training data
            X_train, Y_train = shuffle_data(X_train, Y_train)

            # Loop over the batches
            for step in range(0, m, batch_size):
                end = step + batch_size
                X_batch = X_train[step:end]
                Y_batch = Y_train[step:end]

                # Train the model on the mini-batch
                _, step_cost, step_accuracy = sess.run([train_op, loss, accuracy],
                                                       feed_dict={x: X_batch, y: Y_batch})

                # Print progress after every 100 steps
                if step % 100 == 0:
                    print("\tStep {}: Cost: {:.6f}, Accuracy: {:.4f}".format(step, step_cost, step_accuracy))

            # Calculate validation metrics after each epoch
            train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            # Print epoch results
            print("After {} epochs:".format(epoch + 1))
            print("\tTraining Cost: {:.6f}, Training Accuracy: {:.4f}".format(train_cost, train_accuracy))
            print("\tValidation Cost: {:.6f}, Validation Accuracy: {:.4f}".format(valid_cost, valid_accuracy))

        # Save the trained model
        saver.save(sess, save_path)

    return save_path
