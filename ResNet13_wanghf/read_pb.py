import sys
import cPickle
sys.path.append("..")
sys.path.append("../..")

import numpy as np
class PbStore(object):
    def __init__(self):
        pass

    def read_pb(self, pb_file_path):
        self.pb_file_path = pb_file_path
        with open(self.pb_file_path, "rb") as f:
            x = f.read()
        return x



if __name__ == "__main__":
    pb_file_path = "./model/frozen_model.pb"
    pb = PbStore()
    pb_data = pb.read_pb(pb_file_path)
    cPickle.dump(pb_data, open('pb_test.pkl', 'wb'), protocol=2)




