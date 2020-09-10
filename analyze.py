from simpleDT import *
from data import *


class FeatureHistogram:
    def __init__(root, len):
        self.root = root
        self.his = np.zeros((len))
    
    def front_digui(node):
        if not node.is_leaf:
            self.his[node.split_index]+=1
        front_digui(node.left_child)
        front_digui(node.right_child)
    
    def count():
        front_digui(self.root)
        return self.his

if __name__ == "__main__":
    data_name = "pima"
    metric = "auc"
    result = PickleLoad('result_' + data_name + '_' + metric + '.pkl')
    for i in result:
        print(i[0])