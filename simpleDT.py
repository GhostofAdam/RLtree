from collections import deque
import math
import numpy as np
import sklearn

# random sample ratio
k = 0.8

def entropyImpurity(label):
    p = (label == 1).sum() / float(label.shape[0])
    if (p == 0 or p == 1):
        entropy = 0.0
    else:
        entropy = -(p * math.log(p, 2) + (1 - p) * math.log(1 - p, 2))
    return entropy

def giniImpurity(label):
    if (label.shape[0] <= 1):
        return 0.0

    gini_impurity = 1.0
    p_one = label.sum() / float(label.shape[0])
    p_zero = 1.0 - p_one
    gini_impurity -= (p_one ** 2 + p_zero ** 2)
    return gini_impurity


def TotalGain(root):
    if (root.is_leaf):
        return 0.0
    
    left_child = root.left_child
    right_child = root.right_child
    current_gain = root.impurity - (left_child.impurity * left_child.data_idx.shape[0] + \
        right_child.impurity * right_child.data_idx.shape[0]) / float(root.data_idx.shape[0])
    left_subtree_gain = TotalGain(left_child)
    right_subtree_gain = TotalGain(right_child)
    return (current_gain + left_subtree_gain + right_subtree_gain)


class Node():

    def __init__(self, is_leaf=False, classification=None, data_idx=None, index=None, split_index=None, \
                 split_value=None, parent=None, left_child=None, right_child=None, height=None):
        self.is_leaf = is_leaf
        self.classification = classification
        self.data_idx = data_idx
        self.index = index
        self.split_index = split_index
        self.split_value = split_value
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        self.height = height
        self.impurity = 0.0


    def searchSplit(self, features, label, split_value_list):
        node_feature = features[self.data_idx, self.split_index]
        node_label = label[self.data_idx]
        
        if (len(split_value_list) == 1):
            self.split_value = split_value_list[0]
        else:
            #split_value_list = list(set(node_feature))
            '''
            split_value_array = np.sort(np.array(list(set(node_feature))))
            split_value_array = (split_value_array[:-1] + split_value_array[1:]) / 2.0
            if (split_value_array.shape[0] == 0):
                return False
            '''
            
            split_value_list = list(set(node_feature))
            num = 1600
            length = len(split_value_list)
            if(length > num):
                maximum = max(split_value_list)
                minimum = min(split_value_list)
                d = (maximum - minimum) / float(num)
                new_list = []
                for i in range(num):
                    new_list.append(minimum + 0.5 * d + i * d)
                split_value_list = new_list
                #length = node_feature.shape[0]
                '''
                num = max(200, int(length / 10))
                split_value_list = sorted(split_value_list)
                percentile = int(length / (num + 1))
                new_list = []
                for i in range(1, num + 1):
                    new_list.append(split_value_list[i * percentile])
                split_value_list = new_list
                '''
            
            #max_gain = 0.0
            split_impurity = np.zeros(len(split_value_list))
            
            for i in range(len(split_value_list)):
                value = split_value_list[i]
                left_label = node_label[node_feature <= value]
                right_label = node_label[node_feature > value]
                split_impurity[i] = (giniImpurity(left_label) * left_label.shape[0] \
                                   + giniImpurity(right_label) * right_label.shape[0]) / float(node_label.shape[0])
            idx = np.argmin(split_impurity)
            self.split_value = split_value_list[idx]

        left_sample_num = (node_feature <= self.split_value).sum()
        if (left_sample_num == 0 or left_sample_num == node_label.shape[0]):
            return False
        else:
            return True


    def split(self, features, label):
        node_feature = features[self.data_idx, self.split_index]
        node_label = label[self.data_idx]

        left_label = node_label[node_feature <= self.split_value]
        left_idx = self.data_idx[node_feature <= self.split_value]
        left_child = Node(data_idx=left_idx, index=self.index * 2 + 1, parent=self, height=self.height + 1)
        left_child.impurity = giniImpurity(left_label)
        self.left_child = left_child

        right_label = node_label[node_feature > self.split_value]
        right_idx = self.data_idx[node_feature > self.split_value]
        right_child = Node(data_idx=right_idx, index=self.index * 2 + 2, parent=self, height=self.height + 1)
        right_child.impurity = giniImpurity(right_label)
        self.right_child = right_child

        return left_child, right_child


    def classify(self, label):
        node_label = label[self.data_idx]
        if (node_label.shape[0] == 0):
            print ('zero label')
        else:
            self.classification = (node_label == 1).sum() / float(node_label.shape[0])
        return self.classification


def buildTree(index, data, max_depth, min_samples_leaf, min_impurity_split, shuffle=False):
    
    features = data['X_train']
    label = data['Y_train']

    '''
    if (shuffle):
        size = features.shape[0]
        idx = np.arange(size)
        np.random.shuffle(idx)
        idx = idx[:int(size * k)]
        features = features[idx]
        label = label[idx]
    '''

    root = Node(is_leaf=False, data_idx=np.arange(features.shape[0]), index=0, height=0)
    root.impurity = giniImpurity(label[root.data_idx])
    node_deque = deque()
    node_deque.append(root)

    while (node_deque):
        node = node_deque.popleft()
        if (node.height == max_depth or (node.data_idx).shape[0] <= min_samples_leaf \
            or (node.impurity) <= min_impurity_split):
            node.is_leaf = True
            node.classify(label)
        else:
            node.split_index = index[node.index]
            if (node.searchSplit(features, label, data['split'][node.split_index])):
                left_child, right_child = node.split(features, label)
                node_deque.extend([left_child, right_child])
            else:
                node.is_leaf = True
                node.classify(label)

    return root


def predict(node, features):
    if (node.is_leaf == True):
        return node.classification
    else:
        if (features[node.split_index] <= node.split_value):
            return predict(node.left_child, features)
        else:
            return predict(node.right_child, features)


def test(root, data, test_type, metric='acc', shuffle=False):
    features = data['X_' + test_type]
    label = data['Y_' + test_type]

    '''
    if (shuffle):
        size = features.shape[0]
        idx = np.arange(size)
        np.random.shuffle(idx)
        idx = idx[:int(size * k)]
        features = features[idx]
        label = label[idx]
    '''

    prediction = np.zeros_like(label, dtype='float')
    for i in range(features.shape[0]):
        prediction[i] = predict(root, features[i, :])
    
    if (metric == 'all'):
        return sklearn.metrics.roc_auc_score(label, prediction, average='weighted'), \
               sklearn.metrics.f1_score(label, np.round(prediction), average='weighted'), \
               sklearn.metrics.accuracy_score(label, np.round(prediction))
    elif (metric == 'f1'):
        return sklearn.metrics.f1_score(label, np.round(prediction), average='weighted')
    elif (metric == 'acc'):
        return sklearn.metrics.accuracy_score(label, np.round(prediction))
    elif (metric == 'auc'):
        return sklearn.metrics.roc_auc_score(label, prediction, average='weighted')


def validate(root, data, metric, times):
    features = data['X_valid']
    label = data['Y_valid']

    size = features.shape[0]
    score = 0.0
    for n in range(times):
        idx = np.arange(size)
        np.random.shuffle(idx)
        idx = idx[:int(size * k)]
        sample_features = features[idx]
        sample_label = label[idx]

        prediction = np.zeros_like(sample_label, dtype='float')
        for i in range(sample_features.shape[0]):
            prediction[i] = predict(root, sample_features[i, :])
        if (metric == 'f1'):
            score += sklearn.metrics.f1_score(sample_label, np.round(prediction), average='weighted')
        elif (metric == 'acc'):
            score += sklearn.metrics.accuracy_score(sample_label, np.round(prediction))
        elif (metric == 'auc'):
            score += sklearn.metrics.roc_auc_score(sample_label, prediction, average='weighted')
    score /= float(times)
    return score


def compare(node1, node2):
    if (node1.is_leaf and node2.is_leaf):
        return True
    elif (node1.is_leaf == False and node2.is_leaf == False):
        if (node1.split_index == node2.split_index):
            return (compare(node1.left_child, node2.left_child) and compare(node1.right_child, node2.right_child))
        else:
            return False
    else:
        return False