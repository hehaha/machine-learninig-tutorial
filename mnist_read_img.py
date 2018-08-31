# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import argparse
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


def main(input_path):
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
    images = tf.reshape(images, (300, 300, 4))
    labels = tf.reshape(labels, (1, ))
    images, labels = tf.train.shuffle_batch(
        [images, labels],
        batch_size=10,
        num_threads=2,
        capacity=1000,
        min_after_dequeue=100)
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        images, labels = sess.run([images, labels])
        print images, images.shape
        print to_categorical(labels, 10), labels.shape
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str)
    args = parser.parse_args()
    main(args.i or "/Users/wecash/Desktop/mnist_png/tfrecords/testing")
