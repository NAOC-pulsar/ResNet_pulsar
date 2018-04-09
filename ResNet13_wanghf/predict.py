import os
import sys

sys.path.append("..")
sys.path.append("../..")
import time
import numpy as np
import tensorflow as tf
import cv2

class ResNet_predict():
    def __init__(self, path):
        self.num_classes = 2
        self.image_size = 64
        self.mean = np.array([127.5, 127.5, 127.5])
        self.checkpoint_path = "./tmp/resnet13_64/checkpoints"
        self.predict(path)

    def get_data(self, path):
        img = cv2.imread(path).astype('float64')
        img -= self.mean
        img = np.resize(img, (1, self.image_size, self.image_size, 3))
        return img

    def predict(self, path):
        img = self.get_data(path)
        ckpt = tf.train.get_checkpoint_state('./tmp/resnet13_64/checkpoints/')
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            new_saver.restore(sess, ckpt.model_checkpoint_path)
            tf.get_default_graph().as_graph_def()
            x = sess.graph.get_tensor_by_name('input:0')
            y = sess.graph.get_tensor_by_name('output:0')

            result = sess.run(y, feed_dict={x: img})
            label = np.argmax(result, 1)
            # print("predict label {}, confidence {}%".format(label[0], result[0][label[0]] * 100))
            self.predict = label[0]
            self.confidence = result[0][label[0]]
            self.result = (self.predict, self.confidence)


if __name__ =='__main__':
    path = "/ssd/wanghf/PycharmProjects/ResNet13/dataset/mnist_pic_new/train/0/3384.bmp"
    resnet_pre = ResNet_predict(path)
    result = resnet_pre.result
    print(result)



