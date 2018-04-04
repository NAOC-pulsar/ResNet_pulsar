import numpy as np

class MNISTGenerator():
    def __init__(self):
        self.num_classes = 10
        self.img_size = 28
        self.trainset_num = 60000
        self.testset_num = 10000
        self.train_point = 0
        self.test_point = 0
        self.load_mnist()

    def load_mnist(self, path="dataset/mnist.npz"):
        f = np.load(path)
        self.x_train = f['x_train']
        self.y_train = f['y_train']
        print(self.x_train.shape, self.y_train.shape)
        self.x_test = f['x_test']
        self.y_test = f['y_test']
        f.close()

    def next_train_batch(self, batch_size):
        image = self.x_train[self.train_point: self.train_point + batch_size]
        label = self.y_train[self.train_point: self.train_point + batch_size]

        self.train_point += batch_size


    def reset_point(self):
        self.train_point = 0
        self.test_point = 0

mnist = MNISTGenerator()