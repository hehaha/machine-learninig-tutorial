# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import argparse
import numpy as np
import tensorflow as tf


IMAGE_SHAPE = (300, 300, 4)
FC_SIZE = 512
OUTPUT_NODE = 10
LEARNING_RATE = 0.0001
TRAINING_STEPS = 20
TRAINING_BATCH_SIZE = 32
TESTING_BATCH_SIZE = 128


def test(input_dir):
    model = tf.keras.models.load_model("./cnn_mnist.h5")
    test_list = []
    for i in range(10):
        raw_data = tf.gfile.FastGFile(input_dir + "/" + "%d.png" % i, "r").read()
        encode_img = tf.image.decode_png(raw_data)
        img_matrix = encode_img.eval(session=tf.Session())
        test_list.append(np.expand_dims(img_matrix, 0))
    test_list = np.concatenate(test_list)
    print model.predict(test_list, batch_size=10)


def read(input_path, size):
    reader = tf.TFRecordReader()
    files = tf.train.match_filenames_once(input_path + "/img_[0-9].tfrecords")
    file_queue = tf.train.string_input_producer(files, shuffle=False)
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })
    images = tf.image.decode_png(features["image"])
    labels = tf.cast(features["label"], tf.int32)
    images = tf.reshape(images, IMAGE_SHAPE)
    labels = tf.reshape(labels, (1, ))
    images, labels = tf.train.shuffle_batch(
        [images, labels],
        batch_size=size,
        num_threads=5,
        capacity=1000,
        min_after_dequeue=500)
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        images, labels = sess.run([images, labels])
        coord.request_stop()
        coord.join(threads)
    return images, labels


def main(input_path, test_path):
    images, labels = read(input_path, 60000)
    labels = to_categorical(labels, 10)

    model = keras.Sequential([
        keras.layers.Convolution2D(
            filters=512,
            strides=(2, 2),
            kernel_size=(5, 5),
            input_shape=IMAGE_SHAPE,
            activation="relu",
            bias_initializer=keras.initializers.constant(0.0)),
        keras.layers.MaxPool2D(),
        keras.layers.Convolution2D(
            filters=512,
            kernel_size=(5, 5),
            activation="relu",
            bias_initializer=keras.initializers.constant(0.0)),
        keras.layers.MaxPool2D(),
        keras.layers.Convolution2D(
            filters=256,
            kernel_size=(3, 3),
            activation="relu",
            bias_initializer=keras.initializers.constant(0.0)),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(0.25),
        keras.layers.Convolution2D(
            filters=256,
            kernel_size=(3, 3),
            activation="relu",
            bias_initializer=keras.initializers.constant(0.0)),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(0.25),
        keras.layers.Convolution2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu",
            bias_initializer=keras.initializers.constant(0.0)),
        keras.layers.Convolution2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu",
            bias_initializer=keras.initializers.constant(0.0)),
        keras.layers.Flatten(),
        keras.layers.Dense(
            FC_SIZE,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(0.0001),
            bias_initializer=keras.initializers.constant(0.1)),
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
    model.fit(images, labels, epochs=TRAINING_STEPS, batch_size=TRAINING_BATCH_SIZE)

    test_images, test_labels = read(test_path, 10000)
    test_labels = to_categorical(test_labels, 10)
    print model.evaluate(test_images, test_labels, batch_size=TESTING_BATCH_SIZE)
    model.save("./cnn_mnist.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str)
    parser.add_argument("-t", type=str)
    args = parser.parse_args()
    main(args.i or "/home/hexin/mnist_png/tfrecords/training",
         args.t or "/home/hexin/mnist_png/tfrecords/testing")
    # test("/Users/wecash/Desktop/mnist_test")
