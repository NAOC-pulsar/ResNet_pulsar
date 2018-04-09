# encoding=utf-8
"""
With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset.
Specify the configuration settings at the beginning according to your
problem.
This script was written for TensorFlow 1.0 and come with a blog post
you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""
import os
import sys

sys.path.append("..")
sys.path.append("../..")
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from src.ResNet_master import resnet_model
from src.utils.datagenerator import ImageDataGenerator
from src.utils.utils import mkdirs, plot_confusion_matrix
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

device_id = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = device_id

train_file = "../../path/train.txt"
val_file = "../../path/valid.txt"

# Learning params
learning_rate = 0.0001
num_epochs = 100
batch_size = 100
weight_decay = 0.0002
# Network params
dropout_rate = 0.2
num_classes = 5
image_size = 64

# whether the targets are in the top K predictions.
top_N = 1

checkpoint_path = "../../tmp/face_age_resnet13_64/checkpoints"

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

hps = resnet_model.HParams(batch_size=batch_size,
                           num_classes=num_classes,
                           min_lrn_rate=learning_rate,
                           lrn_rate=0.1,
                           num_residual_units=1,
                           use_bottleneck=False,
                           weight_decay_rate=weight_decay,
                           relu_leakiness=0.1,
                           optimizer='mom')

# Initialize model
model = resnet_model.ResNet(hps, x, y, mode='valid')

# Link variable to model output
y_hat = model.out

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y))

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initalize the data generator seperately for the training and validation set
val_generator = ImageDataGenerator(val_file, shuffle=False, scale_size=(image_size, image_size), nb_classes=num_classes)

# Get the number of training/validation steps per epoch
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)

saver = tf.train.Saver()

checkpoint_file = os.path.join(checkpoint_path, 'model_epoch200.ckpt')
# Start Tensorflow session
with tf.Session() as sess:
    with tf.device('/gpu:' + device_id):
        # tf.initialize_all_variables().run()
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,checkpoint_file)
    #     if not checkpoint_file == '':
    #         print(checkpoint_file)
    #         saver.restore(sess, checkpoint_file)

        t1 = time.time()
        test_acc = 0.
        test_count = 0
        y_batch_predict = np.zeros((batch_size, num_classes))
        y_true = np.zeros((batch_size, 1))
        val_batches_per_epoch = 1
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            valid_loss, valid_acc, valid_y = sess.run([cost, accuracy, y_hat],
                                                      feed_dict={x: batch_tx, y: batch_ty,
                                                                 keep_prob: 1.})
            test_acc += valid_acc
            test_count += 1
        test_acc /= test_count
        t2 = time.time() - t1
        print("Test account {}, cost time {}".format(test_count, t2))
        print("Test image {:.5f}ms per image".format(t2 * 1000 / (val_batches_per_epoch * batch_size)))
        print("Validation Accuracy = {}".format(test_acc))

        valid_y = np.argmax(valid_y, 1)
        for j in range(batch_size):
            y_batch_predict[j][valid_y[j]] = 1

        y_true = np.argmax(batch_ty, 1)
        class_report = classification_report(batch_ty, y_batch_predict)
        confusion_matrix = confusion_matrix(y_true, valid_y)
        print(confusion_matrix)
