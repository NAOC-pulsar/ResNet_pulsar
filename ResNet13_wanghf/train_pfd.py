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
import resnet_model
#from datagenerator import ImageDataGenerator
from pfdGenerator_old import ImageDataGenerator
from utils import mkdirs
from sklearn.metrics import confusion_matrix, f1_score, recall_score, classification_report


# device_id = '0'
#
# os.environ['CUDA_VISIBLE_DEVICES'] = device_id

# train_file = "path/mnist/train.txt"
# val_file = "path/mnist/valid.txt"

train_file = "../../datasets/pfd_data/trainFvPs_shuffle_2.pkl"
val_file = "../../datasets/pfd_data/valFvPs_shuffle_2.pkl"
train_target = "../../datasets/pfd_data/train_target_shuffle_2.pkl"
val_target = "../../datasets/pfd_data/val_target_shuffle_2.pkl"

# train_file = "../datasets/HTRU/trainFvPs_shuffle_2.pkl"
# val_file = "../datasets/HTRU/valFvPs_shuffle_2.pkl"
# train_target = "../datasets/HTRU/train_target_shuffle_2.pkl"
# val_target = "../datasets/HTRU/val_target_shuffle_2.pkl"



image_size = 64

# Learning params
num_epochs = 100
batch_size = 16
learning_rate = 0.0001
weight_decay = 0.0002

# Network params
num_classes = 2

# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "tmp/resnet13_64/tensorboard"
checkpoint_path = "tmp/resnet13_64/checkpoints"
mkdirs(filewriter_path)
mkdirs(checkpoint_path)

# restore_checkpoint = ''
# restore_checkpoint = os.path.join(checkpoint_path, 'epoch_1.data')
ckpt = tf.train.get_checkpoint_state('./tmp/resnet13_64/checkpoints/')
restore_checkpoint = ckpt.model_checkpoint_path
# print(restore_checkpoint)


# arg pars finished ==================================================


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [None, image_size, image_size, 1], name='input')
y = tf.placeholder(tf.float32, [None, num_classes])

hps = resnet_model.HParams(batch_size=batch_size,
                           num_classes=num_classes,
                           num_residual_units=1,
                           use_bottleneck=False,
                           relu_leakiness=0.1,
                           weight_decay_rate=weight_decay)

# Initialize model
model = resnet_model.ResNet(hps, x, y)

# Link variable to model output
predict = model.out
output = tf.nn.softmax(predict, name='output')

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    cost += model._decay()

# freeze_var = ['unit_1_0/sub1/conv1/DW:0', 'unit_1_0/sub1/conv1/beta:0', ]

# List of trainable variables of the layers we want to train
var_list = [
    v for v in tf.trainable_variables()
]

for v in var_list:
    print("output all the trainable variable")
    print(v)

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(cost, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', cost)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# # Initalize the data generator seperately for the training and validation set
# train_generator = ImageDataGenerator(train_file, scale_size=(image_size, image_size), nb_classes=num_classes)
# val_generator = ImageDataGenerator(val_file, scale_size=(image_size, image_size), nb_classes=num_classes)
# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file, train_target, shuffle=True, scale_size=(image_size, image_size), nb_classes=num_classes)
val_generator = ImageDataGenerator(val_file, val_target, shuffle=True, scale_size=(image_size, image_size), nb_classes=num_classes)



# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)

# Start Tensorflow session
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    # if restore_checkponit is '' use ariginal weights, else use checkponit
    if not restore_checkpoint == '':
        saver.restore(sess, restore_checkpoint)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard :tensorboard --logdir {} --host localhost --port 6006".format(datetime.now(),
                                                                                             filewriter_path))
    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("Epoch number: {}/{}".format(epoch + 1, num_epochs))

        step = 1

        while step < train_batches_per_epoch:
            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)
            # And run the training op
            feed_dict = {x: batch_xs, y: batch_ys}
            sess.run(train_op, feed_dict=feed_dict)

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                loss, acc, s = sess.run([cost, accuracy, merged_summary], feed_dict=feed_dict)
                writer.add_summary(s, epoch * train_batches_per_epoch + step)
                print("Iter {}/{}, training mini-batch loss = {:.5f}, training accuracy = {:.5f}".format(
                    step * batch_size, train_batches_per_epoch * batch_size, loss, acc))
                # val_generator.reset_pointer()

            step += 1

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        v_loss = 0.
        v_acc = 0.
        count = 0
        t1 = time.time()
        y_predict = np.zeros((batch_size, num_classes))
        # conf_matrix = np.ndarray((num_classes, num_classes))
        print("valid batchs {}".format(val_batches_per_epoch))
        for i in range(val_batches_per_epoch):
            batch_validx, batch_validy = val_generator.next_batch(batch_size)
            valid_loss, valid_acc, valid_out = sess.run([cost, accuracy, output],
                                                        feed_dict={x: batch_validx, y: batch_validy})

            v_loss += valid_loss
            v_acc += valid_acc
            count += 1

            y_true = np.argmax(batch_validy, 1)
            y_pre = np.argmax(valid_out, 1)
            # print(y_true)
            # print(y_pre)
            # for k in range(batch_size):
            #     if not (y_pre[k] == 0 or y_pre[k] == 1):
            #         y_pre[k] = 0

            if i == 0:
                conf_matrix = confusion_matrix(y_true, y_pre)
            else:
                conf_matrix += confusion_matrix(y_true, y_pre)
            # print(i, conf_matrix)
        v_loss /= count
        v_acc /= count
        t2 = time.time() - t1
        print("Validation loss = {:.4f}, acc = {:.4f}".format(v_loss, v_acc))
        print("Test image {:.4f}ms per image".format(t2 * 1000 / (val_batches_per_epoch * batch_size)))
        print(conf_matrix)

        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()

        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'epoch_' + str(epoch))
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
