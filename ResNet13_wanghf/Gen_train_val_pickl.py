import pickle
import numpy as np
from PIL import Image

def save_pickle(file,filesavepath):
    with open(filesavepath, 'wb') as filepath:
        pickle.dump(file, filepath)
        filepath.close()
    print ('save success!')

def load_pickle(picklepath):
    with open(picklepath, "rb") as file:
        data = pickle.load(file, encoding='iso-8859-1')

    return data

if __name__ == '__main__':
    train_file = "../datasets/pfd_data/valFvps.pkl"
    target_file = "../datasets/pfd_data/val_target.pkl"
    images = []
    labels = []
    # arr_img = load_pickle(train_file)
    target = load_pickle(target_file)
    # images = np.array(arr_img)
    labels = np.array(target)
    ### show image
    # from pylab import *
    # for i in range(10):
    #     img = images[i]
    #     print(img.shape)
    #     imshow(img)
    #     show()
    ### show image

    for i in range(10):
        print(labels[i])
        # train_image = images[:np.int(0.6*len(images))]
    # val_image = images[np.int(0.6*len(images)):]
    # train_label = labels[:np.int(0.6*len(labels))]
    # val_label = labels[np.int(0.6*len(labels)):]
    # save_pickle(train_image, 'trainFvPs.pkl')
    # save_pickle(val_image, 'valFvPs.pkl')
    # save_pickle(train_label, 'train_target.pkl')
    # save_pickle(val_label, 'val_target.pkl')
