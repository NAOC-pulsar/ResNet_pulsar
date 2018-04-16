# -*- coding: utf-8 -*-
# @Time    : 2018/4/8 19:17
# @Author  : Wanghf
# @Email   : dzuwhf@163.com
# @File    : PB_model.py

import tensorflow as tf
import shutil
import os.path
from tensorflow.python.framework import graph_util
import sys
sys.path.append("..")
sys.path.append("../..")
# output_graph = './model/add_model.pb'
MODEL_DIR = "./model/"
MODE_NAME = "frozen_model.pb"
checkpoint_path = "./model"

def freeze_graph(model_folder):
    # input_checkpoint = model_folder
    output_graph = os.path.join(MODEL_DIR, MODE_NAME)

    # output_node_names = "logit"
    ckpt = tf.train.get_checkpoint_state(model_folder)
    restore_checkpoint = ckpt.model_checkpoint_path
    saver = tf.train.import_meta_graph(restore_checkpoint + '.meta')

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, restore_checkpoint)

        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            ["output"]
        )

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))

    for op in graph.get_operations():
        print("name:", op.name)
    print("success!")

    #下面是用于测试，读取pd模型，显示变量名字
    # graph = load_graph("./model/frozen_model.pb")
    # for op in graph.get_operations():
    #     print("name1111:", op.name)
    #
    # pred = graph.get_tensor_by_name('prefix/inputs_placeholder:0')
    # print(pred)
    # temp = graph.get_tensor_by_name('prefix/predictions:0')
    # print(temp)


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

if __name__ == '__main__':
    freeze_graph("model")