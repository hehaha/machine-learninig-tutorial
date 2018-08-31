# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import os
import sys


PATH = "/Users/wecash/Desktop"
DIR_PATH = PATH + "/mnist_png"
DATA_SET = ["training"]


def main(file_path):
    try:
        os.mkdir(DIR_PATH)
    except:
        pass
    for dataset in DATA_SET:
        taget_dir = DIR_PATH + "/" + dataset
        # os.mkdir(taget_dir)
        origin_dir = file_path + "/" + dataset
        dir_list = filter(lambda n: not n.startswith("."), os.listdir(origin_dir))

        for dir_name in dir_list:
            if dir_name in ["9", "0", "6", "7", "1", "5", "4", "3", "2"]:
                continue
            target_num_dir = taget_dir + "/" + dir_name
            os.mkdir(target_num_dir)
            origin_num_dir = origin_dir + "/" + dir_name
            file_list = filter(lambda n: not n.startswith("."), os.listdir(origin_num_dir))
            index = 0
            img_file_list = file_list[:500]
            while img_file_list:
                with tf.Session() as sess:
                    for img_name in img_file_list:
                        print dir_name, img_name
                        raw_data = tf.gfile.FastGFile(origin_num_dir + "/" + img_name, "r").read()
                        img_data = tf.image.decode_png(raw_data)
                        resized = tf.image.resize_images(img_data, [300, 300], method=1)
                        encode_img = tf.image.encode_png(resized)
                        with tf.gfile.GFile(target_num_dir + "/" + img_name, "wb") as f:
                            f.write(encode_img.eval())
                index += 500
                img_file_list = file_list[index: index + 500]
                sys.stdout.flush()


if __name__ == "__main__":
    main("/Users/wecash/Desktop/mnist_png-master")
