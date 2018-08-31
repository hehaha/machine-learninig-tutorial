# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np


INPUT_NODE = 4
OUTPUT_NODE = 3
LAYER1_NODE = 20
LAYER2_NODE = 20

LAMADA = 0.0001
MOVING_AVERAGE_DECAY = 0.999
TRAINING_STEPS = 2000
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_BASE = 0.1


def get_weight(shape, Lamada):
    weights = tf.get_variable(
        "weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if Lamada:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(Lamada)(weights))
    return weights


def inference(input_tensor, avg_class, reuse):
    with tf.variable_scope("layer1", reuse=reuse):
        weights1 = get_weight([INPUT_NODE, LAYER1_NODE], LAMADA)
        biases1 = tf.get_variable("bias", [LAYER1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
    with tf.variable_scope('layer2', reuse=reuse):
        weights2 = get_weight([LAYER1_NODE, LAYER2_NODE], LAMADA)
        biases2 = tf.get_variable("bias", [LAYER2_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
    with tf.variable_scope('output_layer', reuse=reuse):
        weights3 = get_weight([LAYER2_NODE, OUTPUT_NODE], LAMADA)
        biases3 = tf.get_variable("bias", [OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
    if not avg_class:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        layer2 = tf.nn.relu(tf.matmul(layer1, weights2) + biases2)
        output = tf.matmul(layer2, weights3) + biases3
    else:
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        layer2 = tf.nn.relu(
            tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2))
        output = tf.matmul(layer1, avg_class.average(weights3)) + avg_class.average(biases3)
    return output


def main(argv=None):
    x = tf.placeholder(tf.float32, shape=(None, INPUT_NODE), name='x_input')
    y_ = tf.placeholder(tf.int64, shape=(None, ), name='y_input')

    y = inference(x, None, False)
    global_step = tf.Variable(0, trainable=False)

    # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # average_y = inference(x, variable_averages, True)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)
    loss = tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               1,
                                               LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)
    # with tf.control_dependencies([train_step, variable_averages_op]):
    #     train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        test_matrix = np.loadtxt(
            '/Users/wecash/.keras/datasets/iris_test.csv', skiprows=1, delimiter=',')
        features = test_matrix[:, : 4]
        labels = test_matrix[:, 4].astype(np.int)
        test_feed = {x: features, y_: labels}

        matrix = np.loadtxt('/Users/wecash/Desktop/iris_training.csv', skiprows=1, delimiter=",")

        for i in range(TRAINING_STEPS):
            np.random.shuffle(matrix)
            features = matrix[:, : 4]
            labels = matrix[:, 4].astype(np.int)
            train_feed = {x: features[:100], y_: labels[:100]}
            _, loss_value, step = sess.run(
                [train_step, loss, global_step], feed_dict=train_feed)
            if i % 99 == 0:
                validate_feed = {x: features[101:], y_: labels[101:]}
                print(
                    "After %d training step(s), loss on training batch is %g." % (step, loss_value))
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print (
                    "After %d training step(s),validation accuracy using average model is %g " % (
                        step, validate_acc))
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print(
                    "After %d training step(s) testing accuracy using average model is %g" % (
                        step, test_acc))


if __name__ == "__main__":
    tf.app.run()
