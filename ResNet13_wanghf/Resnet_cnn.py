import os
import sys
import pickle
sys.path.append("..")
sys.path.append("../..")

from datetime import datetime
import numpy as np
import tensorflow as tf
import resnet_model
# from datagenerator import ImageDataGenerator
from pfdGenerator import ImageDataGenerator
from utils import mkdirs

class ResNet_CNN(object):
    '''
    def __init__(self, image_size, num_epoch, batch_size, learning_rate,
                 weight_decay, num_classes, filewriter_path, checkpoint_path,
                 num_residual_units, relu_leakiness=0.1, is_bottleneck=True, is_restore=True):
    '''
    def __init__(self, image_size, num_epoch, batch_size, learning_rate,
                 weight_decay, num_classes, checkpoint_path,
                 num_residual_units, relu_leakiness=0.1, is_bottleneck=False, is_restore=True):
        self.image_size = image_size
        self.num_epochs = num_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.display_step = 20 # Display training procedure
        #self.filewriter_path = filewriter_path # Display tensorboard
        #mkdirs(self.filewriter_path)
        self.checkpoint_path = checkpoint_path
        mkdirs(self.checkpoint_path)
        self.num_residual_units = num_residual_units
        self.relu_leakiness = relu_leakiness
        self.is_bottlneck = is_bottleneck
        if is_restore:               # Whether to restore the checkpoint
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
            self.restore_checkpoint = ckpt.model_checkpoint_path
        else:
            self.restore_checkpoint = ''

    def load_pickle(self, picklepath):
        with open(picklepath, "rb") as file:
            #data = pickle.load(file, encoding='iso-8859-1')
            data = pickle.load(file)

        return data

    def read_class_list(self, class_list, target_list):
        """
        Scan the image file and get the image paths and labels
        """
        self.images = []
        self.labels = []
        arr_img = self.load_pickle(class_list)
        target = self.load_pickle(target_list)
        self.labels = np.array(target)
        new_img = np.array(arr_img)

        for i in range(self.labels.size):
            data_FvP = np.zeros([64, 64, 1])
            data_FvP[:, :, 0] = new_img[i]
            # data_FvP[:, :, 1] = new_img[i]
            # data_FvP[:, :, 2] = new_img[i]
            self.images.append(data_FvP)

            # store total number of data
        self.data_size = len(self.labels)

    def fit(self, X_train, Y_train):
        x = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1], name='input')
        y = tf.placeholder(tf.float32, [None, self.num_classes])

        hps = resnet_model.HParams(batch_size=self.batch_size,
                                   num_classes=self.num_classes,
                                   num_residual_units=self.num_residual_units,
                                   use_bottleneck=self.is_bottlneck,
                                   relu_leakiness=self.relu_leakiness,
                                   weight_decay_rate=self.weight_decay)
        model = resnet_model.ResNet(hps, x, y)
        predict = model.out
        output = tf.nn.softmax(predict, name='output')

        with tf.name_scope("cross_ent"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
            cost += model._decay()

        var_list = [v for v in tf.trainable_variables()]

        with tf.name_scope("train"):
            gradients = tf.gradients(cost, var_list)
            gradients = list(zip(gradients, var_list))
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
            train_op = optimizer.apply_gradients(grads_and_vars=gradients)


        with tf.name_scope("accuracy"):
            prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))


        saver = tf.train.Saver()
        #Initialize the data generator seperately for the training set,didn't initialize validation set
        train_generator = ImageDataGenerator(X_train, Y_train, shuffle=True, scale_size=(self.image_size, self.image_size), nb_classes=self.num_classes)
        # Get the number of training steps per epoch
        train_batches_per_epoch = np.floor(self.data_size / self.batch_size).astype(np.int16)

        # Start Tensorflow session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            #writer.add_graph(sess.graph)

            if not self.restore_checkpoint == '':
                saver.restore(sess, self.restore_checkpoint)

            print("{} Start training...".format(datetime.now()))
            #print("{} Open Tensorboard :tensorboard --logdir {} --host localhost --port 6006".format(datetime.now(),self.filewriter_path))
            for epoch in range(self.num_epochs):
                step = 1
                while step < train_batches_per_epoch:
                    # Get a batch of images and labels
                    batch_xs, batch_ys = train_generator.next_batch(self.batch_size)
                    # And run the training op
                    feed_dict = {x: batch_xs, y: batch_ys}
                    sess.run(train_op, feed_dict=feed_dict)

                    # Generate summary with the current batch of data and write to file
                    if step % self.display_step == 0:
                        # loss, acc, s = sess.run([cost, accuracy, merged_summary], feed_dict=feed_dict)
                        loss, acc = sess.run([cost, accuracy], feed_dict=feed_dict)
                        #writer.add_summary(s, epoch * train_batches_per_epoch + step)
                        print("Iter {}/{}, training mini-batch loss = {:.5f}, training accuracy = {:.5f}".format(
                            step * self.batch_size, train_batches_per_epoch * self.batch_size, loss, acc))
                    step += 1
                train_generator.reset_pointer()
        '''
        # save checkpoint
        train_generator.reset_pointer()
        print("{} Saving checkpoint of model...".format(datetime.now()))
        # save checkpoint of the model
        checkpoint_name = os.path.join(self.checkpoint_path, 'epoch_' + str(epoch))
        saver.save(sess, checkpoint_name)
        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))   '''
    def predict(self, predict_X):
        ckpt = tf.train.get_checkpoint_state('tmp/resnet13_64/checkpoints/')
        imgs = predict_X
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            new_saver.restore(sess, ckpt.model_checkpoint_path)
            tf.get_default_graph().as_graph_def()
            x = sess.graph.get_tensor_by_name('input:0')
            y = sess.graph.get_tensor_by_name('output:0')
            result = sess.run(y, feed_dict={x: imgs})
            label = np.argmax(result, 1)
            print(label)


rn = ResNet_CNN(
    image_size=64,
    num_epoch=100,
    batch_size=16,
    learning_rate=0.0001,
    weight_decay=0.0002,
    num_classes=2,
    #filewriter_path="tmp/resnet13_64/tensorboard",
    checkpoint_path="tmp/resnet13_64/checkpoints",
    num_residual_units=1,
    relu_leakiness=0.1,
    is_bottleneck=False,
    is_restore=True
)
train_file = "../datasets/pfd_data/trainFvPs_shuffle_2.pkl"
train_target = "../datasets/pfd_data/train_target_shuffle_2.pkl"

rn.read_class_list(train_file, train_target)
#print(rn.labels)
rn.fit(rn.images, rn.labels)
#rn.predict(rn.images)
