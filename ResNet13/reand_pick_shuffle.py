
import numpy as np
import pickle
class PickleShuffle():
    def __init__(self, pfdfile, targetfile):
        self.pfdfile = pfdfile
        self.targetfile = targetfile
        self.read_pickle_file(self.pfdfile,self.targetfile)
        # self.shuffle_data()
        # self.save_pickle_file()

    def save_pickle(self,file, filesavepath):
        with open(filesavepath, 'wb') as filepath:
            pickle.dump(file, filepath)
            filepath.close()
        print('save success!')

    def load_pickle(self, picklepath):
        with open(picklepath, "rb") as file:
            data = pickle.load(file, encoding='iso-8859-1')
        return data

    def read_pickle_file(self, pfdfile, targetfile):
        self.images = []
        self.labels = []
        arr_img = self.load_pickle(pfdfile)
        target = self.load_pickle(targetfile)
        self.images = np.array(arr_img)
        self.labels = np.array(target)
        self.datasize = len(self.labels)

    def shuffle_data(self):
        images = self.images
        labels = self.labels
        self.images = []
        self.labels = []
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def save_pickle_file(self):
        train_pfd_shuffle = self.images[:np.int(0.6*len(self.labels))]
        val_pfd_shuffle = self.images[np.int(0.6*len(self.labels)):]
        train_label_shuffle = self.labels[:np.int(0.6*len(self.labels))]
        val_label_shuffle = self.labels[np.int(0.6*len(self.labels)):]
        self.save_pickle(train_pfd_shuffle, 'trainFvPs_shuffle.pkl')
        self.save_pickle(val_pfd_shuffle, 'valFvPs_shuffle.pkl')
        self.save_pickle(train_label_shuffle, 'train_target_shuffle.pkl')
        self.save_pickle(val_label_shuffle, 'val_target_shuffle.pkl')

train_file = "../datasets/pfd_data/trainFvPs_shuffle.pkl"
target_file = "../datasets/pfd_data/train_target_shuffle.pkl"

ps = PickleShuffle(train_file, target_file)
# ps.shuffle_data()
# ps.save_pickle_file()

