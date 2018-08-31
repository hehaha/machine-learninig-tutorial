# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import argparse


IMAGE_SHAPE = (300, 300, 4)
CONV1_SIZE = 5
CONV1_DEEP = 16
CONV2_SIZE = 5
CONV2_DEEP = 32
FC_SIZE = 512
OUTPUT_NODE = 10
LEARNING_RATE = 0.01
TRAINING_STEPS = 10
TRAIN_BATCH_SIZE = 10


def read(input_path):
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
        batch_size=100,
        num_threads=5,
        capacity=1000,
        min_after_dequeue=200)
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        images, labels = sess.run([images, labels])
        coord.request_stop()
        coord.join(threads)
    return images, labels


def main(input_path):
    images, labels = read(input_path)
    labels = to_categorical(labels, 10)

    model = keras.Sequential([
        keras.layers.Convolution2D(
            filters=CONV1_DEEP,
            kernel_size=(CONV1_SIZE, CONV1_SIZE),
            input_shape=IMAGE_SHAPE,
            activation="relu",
            bias_initializer=keras.initializers.constant(0.0)),
        keras.layers.MaxPool2D(),
        keras.layers.Convolution2D(
            filters=CONV2_DEEP,
            kernel_size=(CONV2_SIZE, CONV2_SIZE),
            activation="relu",
            bias_initializer=keras.initializers.constant(0.0)),
        keras.layers.MaxPool2D(),
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
    model.fit(images, labels, epochs=TRAINING_STEPS, batch_size=TRAIN_BATCH_SIZE)

    raw_data = tf.gfile.FastGFile("/Users/wecash/Desktop/59869_300.png", 'r').read()
    img_data = tf.image.decode_png(raw_data)
    img_data = img_data.eval(session=keras.backend.get_session()).reshape(1, 300, 300, 4)
    print model.predict(img_data, batch_size=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str)
    args = parser.parse_args()
    main(args.i or "/Users/wecash/Desktop/mnist_png/tfrecords/testing")
