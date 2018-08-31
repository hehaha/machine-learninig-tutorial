# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


INPUT_NODE = 784
LAYER1_NODE = 500
OUTPUT_NODE = 10
LEARNING_RATE = 0.01
TRAINING_STEPS = 100
BATCH_SIZE = 10


def main(mnist):
    model = keras.Sequential([
        keras.layers.Dense(
            LAYER1_NODE,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(0.0001),
            input_shape=(INPUT_NODE, ),
            bias_initializer=keras.initializers.random_normal(stddev=0.1)),
        keras.layers.Dense(
            OUTPUT_NODE,
            activation="softmax",
            kernel_regularizer=keras.regularizers.l2(0.0001),
            bias_initializer=keras.initializers.random_normal(stddev=0.1)),
    ])
    model.compile(
        optimizer=tf.train.RMSPropOptimizer(LEARNING_RATE),
        loss=keras.losses.categorical_crossentropy,
        metrics=[keras.metrics.categorical_accuracy])
    train_labels = mnist.train.labels
    train_images = mnist.train.images
    validation_labels = mnist.validation.labels
    model.fit(
        train_images, train_labels, epochs=TRAINING_STEPS, batch_size=BATCH_SIZE,
        validation_data=(mnist.validation.images, validation_labels))
    raw_data = tf.gfile.FastGFile("/Users/wecash/Desktop/59869.png", 'r').read()
    img_data = tf.image.decode_png(raw_data)
    img_data = img_data.eval(session=keras.backend.get_session()).reshape(1, INPUT_NODE)
    print model.predict(img_data, batch_size=1)


if __name__ == "__main__":
    mnist = input_data.read_data_sets(
        "/Users/wecash/Desktop/Tensorflow-master/MNIST_data/", one_hot=True)
    main(mnist)
