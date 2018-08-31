# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import argparse
import tensorflow as tf
import os
import threading


DIR_NAME_PATH = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image_batch(dirpath, num, output_path):
    file_list = filter(lambda n: not n.startswith("."), os.listdir(dirpath))
    writer = tf.python_io.TFRecordWriter(output_path + "/img_%d.tfrecords" % num)
    for f in file_list:
        print f, num
        file_name = dirpath + "/" + f
        image = tf.gfile.FastGFile(file_name, 'r').read()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(image),
            'label': _int64_feature(num)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def main(input_path, output_path):
    coord = tf.train.Coordinator()
    threads = []
    for dir_name in DIR_NAME_PATH:
        inputdir = input_path + "/" + dir_name
        t = threading.Thread(target=load_image_batch, args=(inputdir, int(dir_name), output_path))
        t.start()
        threads.append(t)
    coord.join(threads)
    print "finish!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", type=str)
    parser.add_argument("-i", type=str)
    args = parser.parse_args()
    main(args.i, args.o)
