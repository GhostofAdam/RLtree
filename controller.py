import matplotlib.pyplot as plt
import numpy as np
from simpleDT import *
from data import *
import math

import graphviz
from progressbar import *


# hyperparameters
# for decision tree
min_impurity_split = 0.0
# for reward calculation
k = 0.9
record_num = 20


class Config():

    def __init__(self, option_size=None, hidden_size=None, tree_depth=None, lr=0.1, min_samples_leaf=1, iterations=None):
        self.option_size = option_size

        self.hidden_size = hidden_size
        self.tree_depth = tree_depth
        self.seq_length = (2 ** tree_depth) - 1
        self.learning_rate = lr
        self.min_samples_leaf = min_samples_leaf
        self.iterations = iterations

def calReward(root, data, metric):
    #valid_acc = test(root, data, 'valid', metric=metric, shuffle=False)
    valid_acc = validate(root, data, metric, 5)
    train_acc = test(root, data, 'train', metric=metric, shuffle=False)
    test_acc = test(root, data, 'test', metric=metric, shuffle=False)

    global temp_train, temp_valid, temp_test
    temp_train = train_acc
    temp_valid = valid_acc
    temp_test = test_acc

    return (2 * valid_acc * train_acc / (valid_acc + train_acc))


def feedback(index, data, config, metric):
    global EMA, best_result, best_tree
    root = buildTree(index, data, config.tree_depth, config.min_samples_leaf, min_impurity_split)
    reward = calReward(root, data, metric)
    
    pos = (best_result < reward).sum()
    if (pos != 0):
        duplicate = False
        for i in range(pos, record_num):
            if (reward == best_result[i]):
                if (compare(root, best_tree[i])):
                    duplicate = True
                    break
            else:
                break
        if (duplicate == False):
            best_result = np.insert(best_result, pos, reward)
            best_result = np.delete(best_result, 0)
            best_tree.insert(pos, root)
            del best_tree[0]

    reward -= EMA
    EMA += (1 - k) * reward
    return reward

'''
def RandomForest(tree_list, data, tree_num=None):
    if (tree_num == None):
        tree_num = len(tree_list)
    features = data['X_test']
    label = data['Y_test']
    prediction = np.zeros([label.shape[0], tree_num], dtype='int')
    final_prediction = np.zeros_like(label)
    for j in range(tree_num):
        root = tree_list[j]
        for i in range(features.shape[0]):
            prediction[i, j] = predict(root, features[i, :])
    for i in range(label.shape[0]):
        final_prediction[i] = np.argmax(np.bincount(prediction[i, :]))
    accuracy = (final_prediction == label).sum() / float(label.shape[0])
    return accuracy
'''

class Controller():
    
    def __init__(self, config):

        # controller config
        self.feature_constrain = config.feature_constrain

        self.option_size = config.option_size

        self.hidden_size = config.hidden_size
        self.seq_length = config.seq_length
        self.lr = config.learning_rate

        # RNN parameters
        self.Wxh = np.random.randn(self.hidden_size, self.option_size) * 0.08 # input to hidden
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.08 # hidden to hidden
        self.Why = np.random.randn(self.option_size, self.hidden_size) * 0.08 # hidden to output
        self.bh = np.zeros((self.hidden_size, 1)) # hidden bias
        self.by = np.zeros((self.option_size, 1)) # output bias
        self.x_start = np.zeros([self.option_size, 1])


    def backPropagation(self, data, config, metric):

        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.zeros([self.hidden_size, 1])
        xs[0] = self.x_start
        prediction_index = np.zeros(self.seq_length, dtype='int')

        # forward propagation
        for t in range(self.seq_length):
            if (t > 0):
                xs[t] = np.zeros([self.option_size, 1])
                xs[t][index] = 1
            #hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)

            if t % 2 == 0:
                hs[t] = np.maximum(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, -hs[int(t/2)-1]) + self.bh, 0.)
            else:
                hs[t] = np.maximum(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[int((t-1)/2)]) + self.bh, 0.)

            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t] - ys[t].max()) / np.sum(np.exp(ys[t] - ys[t].max()))
            index = np.random.choice(range(self.option_size), p=ps[t].ravel())
            prediction_index[t] = index

        reward = feedback(prediction_index, data, config, metric)

        # backward propagation
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        dx_start = np.zeros_like(self.x_start)

        #decay_coef = 1.0
        #alpha = 0.0015
        for t in reversed(range(self.seq_length)):
            dy = -np.copy(ps[t])
            dy[prediction_index[t]] += 1

            '''
            # entropy penalty
            temp = np.dot(ps[t].T, np.log(ps[t]) + 1)
            dy_entropy = alpha * (temp - np.log(ps[t]) - 1) * ps[t]
            dy += dy_entropy
            '''

            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot((self.Why).T, dy) + dhnext
            #dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
            dhraw = (hs[t] > 0).astype(float) * dh # backprop through relu nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)

            if t % 2 == 0:
                dWhh += np.dot(dhraw, -hs[int(t/2)-1].T)
            else:
                dWhh += np.dot(dhraw, hs[int((t-1)/2)].T)


            dhnext = np.dot((self.Whh).T, dhraw)
        dx_start = np.dot((self.Wxh).T, dhraw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby, dx_start]:
            dparam *= reward
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
            #np.clip(dparam, -1., 1., out=dparam) # clip to mitigate exploding gradients

        return dWxh, dWhh, dWhy, dbh, dby, dx_start, reward


    def sample(self):
        
        x = self.x_start
        h = np.zeros([self.hidden_size, 1])
        ixes = []
        for t in range(self.seq_length):
            #h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            h = np.maximum(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh, 0.)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y - y.max()) / np.sum(np.exp(y - y.max()))
            ix = np.random.choice(range(self.option_size), p=p.ravel())
            x = np.zeros([self.option_size, 1])
            x[ix] = 1
            ixes.append(ix)
        return ixes


    def maxSample(self):
        
        x = self.x_start
        h = np.zeros([self.hidden_size, 1])
        ixes = []
        for t in range(self.seq_length):
            #h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            h = np.maximum(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh, 0.)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y - y.max()) / np.sum(np.exp(y - y.max()))
            ix = np.argmax(p)
            x = np.zeros([self.option_size, 1])
            x[ix] = 1
            ixes.append(ix)
        return ixes


    def train(self, data, config, data_name, epoch, metric, iterations=200, batch_size=1):

        # memory variables for ADAM
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)
        mx_start = np.zeros_like(self.x_start)

        vWxh, vWhh, vWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        vbh, vby = np.zeros_like(self.bh), np.zeros_like(self.by)
        vx_start = np.zeros_like(self.x_start)

        beta1 = 0.9
        beta2 = 0.999

        n = 0
        iterations /= batch_size
        sample_record = np.zeros([3, int(iterations + 1)])
        top_model = {'valid':{'auc':[], 'f1':[], 'acc':[]}, 'test':{'auc':[], 'f1':[], 'acc':[]}}
        pbar = ProgressBar().start()
        while (n <= iterations):
            pbar.update(int((n / iterations) * 100))

            # forward seq_length characters through the net and fetch gradient
            dWxh, dWhh, dWhy, dbh, dby, dx_start, reward = self.backPropagation(data, config, metric)
            # perform parameter update with Adagrad
            for param, dparam, m, v in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by, self.x_start], \
                                          [dWxh, dWhh, dWhy, dbh, dby, dx_start], \
                                          [mWxh, mWhh, mWhy, mbh, mby, mx_start], \
                                          [vWxh, vWhh, vWhy, vbh, vby, vx_start]):
                
                # ADAM
                m = beta1 * m + (1.0 - beta1) * dparam
                v = beta2 * v + (1.0 - beta2) * (dparam ** 2)
                m_hat = m / (1.0 - beta1 ** (n + 1))
                v_hat = v / (1.0 - beta2 ** (n + 1))
                param += self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                
                # Adagrad
                #param += self.lr * dparam / np.sqrt(m + 1e-8)
            
            # record
            sample_record[:, n] = np.array([temp_train, temp_valid, temp_test])

            if (n % (iterations / 10) == 0):

                print (n, 'iterations finished')

                # converge model
                '''
                index = self.maxSample()
                tree = buildTree(index, data, config.tree_depth, config.min_samples_leaf, min_impurity_split)
                '''
                # top-1 model
                tree = best_tree[-1]
                top_model = score(tree, data, top_model)

            n += 1 # iteration counter

        pbar.finish()

        return top_model, sample_record


def score(tree, data, result):
    for set_type in ['valid', 'test']:
        score = test(tree, data, set_type, metric='all', shuffle=False)
        result[set_type]['auc'].append(score[0])
        result[set_type]['f1'].append(score[1])
        result[set_type]['acc'].append(score[2])
    return result


def run(data_name, epoch, metric):
    data = PickleLoad(data_name + '.pkl')
    if (data_name == 'pima'):

        config = Config(option_size=8, hidden_size=32, tree_depth=4, lr=3e-4, min_samples_leaf=10, iterations=3000)

    elif (data_name == 'breast_cancer'):
        config = Config(option_size=30, hidden_size=64, tree_depth=5, lr=4e-4, min_samples_leaf=5, iterations=5000)
    elif (data_name == 'heart'):
        config = Config(option_size=20, hidden_size=64, tree_depth=4, lr=0.001, min_samples_leaf=5, iterations=2000)
    elif (data_name == 'german'):
        config = Config(option_size=24, hidden_size=64, tree_depth=5, lr=0.001, min_samples_leaf=5, iterations=2000)
    elif (data_name == 'adult'):
        config = Config(option_size=64, hidden_size=128, tree_depth=5, lr=0.0002, min_samples_leaf=5, iterations=5000)
    elif (data_name == 'chess'):
        config = Config(option_size=36, hidden_size=128, tree_depth=5, lr=0.0002, min_samples_leaf=5, iterations=5000)
    elif (data_name == 'credit'):

        config = Config(option_size=23, hidden_size=64, tree_depth=5, lr=1e-3, min_samples_leaf=5, iterations=10000)

    elif (data_name == 'HTRU'):
        config = Config(option_size=8, hidden_size=32, tree_depth=4, lr=1e-3, min_samples_leaf=5, iterations=2000)
    '''
    elif (data_name == 'spam'):
        config = Config(option_size=57, hidden_size=128, tree_depth=5, lr=0.001, min_samples_leaf=5, iterations=5000)
    elif (data_name == 'mammo'):
        config = Config(option_size=12, hidden_size=64, tree_depth=5, lr=1e-4, min_samples_leaf=10, iterations=5000)
    elif (data_name == 'australia'):
        config = Config(option_size=39, hidden_size=128, tree_depth=4, lr=1e-4, min_samples_leaf=10, iterations=5000)
    '''

    config.lr = 1e-4
    #config.iterations = 5000
    controller = Controller(config)

    global EMA, best_result, best_tree
    EMA = 0.0
    best_result = np.zeros(record_num)
    best_tree = []

    for i in range(record_num):
        index = controller.sample()
        best_tree.append(buildTree(index, data, config.tree_depth, config.min_samples_leaf, min_impurity_split, config.feature_constrain))
        best_result[i] = calReward(best_tree[-1], data, metric)
        EMA += best_result[i]
    EMA /= float(record_num)
    order = np.argsort(best_result)
    best_result = best_result[order]
    best_tree = [best_tree[order[i]] for i in range(record_num)]

    
    top_model, sample_record = controller.train(data, config, data_name, epoch, metric, iterations=config.iterations, constrain=config.feature_constrain)
    
    
    dot_data = dotgraphTree(best_tree[-1])
    graph = graphviz.Source(dot_data)
    graph.render(data_name+"_RL")

    return top_model, sample_record


def experiment(metric):
    datasets = ['heart', 'german', 'pima', 'breast_cancer']
    m = [5,5,5,5]
    #datasets = ['credit']
    n = 100
    for i in range(len(datasets)):
        data_name = datasets[i]
        print (data_name)
        result = []
        for j in range(m[i]):
            print ('%d rounds finished' %(j))
            result.append(run(data_name, j, metric))
        print (result)
        PickleSave('result_' + data_name + '_' + metric + '.pkl', result)



if __name__ == '__main__':
    #run('chess')
    experiment('acc')
    experiment('f1')
    experiment('auc')