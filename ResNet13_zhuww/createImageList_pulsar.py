# encoding=utf-8
import numpy as np
import os
import sys

sys.path.append("..")
sys.path.append("../..")


def gci(path, file_list, label):
    parents = os.listdir(path)
    for parent in parents:
        child = os.path.join(path, parent)
        if os.path.isdir(child):
            gci(child, file_list)
        else:
            # str = os.path.normpath(child) + ' ' + os.path.split(os.path.dirname(child))[-1]
            str = os.path.normpath(child) + ' ' + label

            # str = os.path.normpath('../' + child)
            file_list.append(str)
    return file_list


def write_file(txt_file, file_list):
    with open(txt_file, 'w') as file:
        np.random.shuffle(file_list)
        for fn in file_list:
            try:
                file.write(os.path.abspath(fn))
                print("wtite path {}".format(fn))
                file.writelines('\n')
            except:
                print("path error {}".format(fn))


def write_file_with_split(train_txt, test_txt, file_list):
    np.random.shuffle(file_list)
    with open(train_txt, 'w') as train_file:
        with open(test_txt, 'w') as test_file:
            len = file_list.__len__()
            split_num = len / 7.
            for i in range(len):
                print(file_list[i])
                if i < split_num:
                    test_file.write(file_list[i])
                    test_file.writelines('\n')
                else:
                    train_file.write(file_list[i])
                    train_file.writelines('\n')


# train_path = "/home1/fsb/dataset/face_age/data/train"
# test_path = '/home1/fsb/dataset/face_age/data/test'

train_path = "../datasets/wang_data/pulsar/2"
test_path = "../datasets/wang_data/no/2"

#train_path = "dataset/mnist_pic/train"
#test_path = "dataset/mnist_pic/test"

if not os.path.exists("./path"):
    os.makedirs("./path")

train_txt = "./path/train.txt"
test_txt = "./path/valid.txt"

file_list_a = []
file_list_b = []
gci(train_path, file_list_a, '1')
gci(test_path, file_list_b, '0')
write_file_with_split(train_txt, test_txt, file_list_a + file_list_b) 
#write_file(train_txt, gci(train_path, file_list_a))
#write_file(test_txt, gci(test_path, file_list_b))
