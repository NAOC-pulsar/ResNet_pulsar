# -*- coding: utf-8 -*-
# @Time    : 2018/4/9 8:57
# @Author  : Wanghf
# @Email   : dzuwhf@163.com
# @File    : predict_by_pb.py

import os
import sys

sys.path.append("..")
sys.path.append("../..")

import numpy as np
import tensorflow as tf
import cv2

class ResNet_predict():
    def __init__(self, path):
        self.num_classes = 2
        self.image_size = 64
        self.mean = np.array([127.5])
        self.pb_path = "model/frozen_model.pb"
        self.predict(path)

    def get_data(self, path):
        img = cv2.imread(path).astype('float64')
        img -= self.mean
        img = np.resize(img, (1, self.image_size, self.image_size, 3))
        imgs0 = np.zeros([1, 64, 64, 1])
        imgs0[0, :, :, 0] = img[0, :, :, 0]
        return imgs0

    def predict(self, path):
        imgs = self.get_data(path)
        pb_file_path = "./model/frozen_model.pb"

        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()

            with open(pb_file_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")

            with tf.Session() as sess:
                # init = tf.global_variable_initializer()
                # sess.run(init)
                input_x = sess.graph.get_tensor_by_name("input:0")
                output_label = sess.graph.get_tensor_by_name("output:0")
                result = sess.run(output_label, feed_dict={input_x: imgs})
                label = np.argmax(result, 1)
                self.result = label

if __name__ == '__main__':
    path = "dataset/1.bmp"
    resnet_pre = ResNet_predict(path)
    result = resnet_pre.result
    print(result)



