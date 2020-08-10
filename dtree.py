import csv
from collections import defaultdict
import pydotplus
import torch 


class DecisionTree:
    """Binary tree implementation with true and false branch. """
    def __init__(self, col=-1, index = -1, value=None, trueBranch=None, falseBranch=None, results=None, summary=None):
        self.col = col
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results # None for nodes, not None for leaves
        self.index = index
        self.summary = summary


def divideSet(rows, column, value):
    splittingFunction = None
    if isinstance(value, int) or isinstance(value, float): # for int and float values
        splittingFunction = lambda row : row[column] >= value
    else: # for strings
        splittingFunction = lambda row : row[column] == value
    list1 = [row for row in rows if splittingFunction(row)]
    list2 = [row for row in rows if not splittingFunction(row)]
    return (list1, list2)


def uniqueCounts(rows):
    results = {}
    for row in rows:
        #response variable is in the last column
        r = row[-1]
        if r not in results: results[r] = 0
        results[r] += 1
    return results


def entropy(rows):
    from math import log
    log2 = lambda x: log(x)/log(2)
    results = uniqueCounts(rows)

    entr = 0.0
    for r in results:
        p = float(results[r])/len(rows)
        entr -= p*log2(p)
    return entr


def gini(rows):
    total = len(rows)
    counts = uniqueCounts(rows)
    imp = 0.0

    for k1 in counts:
        p1 = float(counts[k1])/total
        for k2 in counts:
            if k1 == k2: continue
            p2 = float(counts[k2])/total
            imp += p1*p2
    return imp


def variance(rows):
    if len(rows) == 0: return 0
    data = [float(row[len(row) - 1]) for row in rows]
    mean = sum(data) / len(data)

    variance = sum([(d-mean)**2 for d in data]) / len(data)
    return variance


def growDecisionTreeFrom(rows, i, chosen_features ,evaluationFunction=entropy):
    """Grows and then returns a binary decision tree.
    evaluationFunction: entropy or gini"""
    #print(i)
    
    if len(rows) == 0: return DecisionTree()
    currentScore = evaluationFunction(rows)
    dcY = {'impurity' : '%.3f' % currentScore, 'samples' : '%d' % len(rows)}
    if i > len(chosen_features):
        return DecisionTree(results=uniqueCounts(rows),summary=dcY)
    # bestGain = 0.0
    # bestAttribute = None
    # bestSets = None

    columnCount = len(rows[0]) - 1  # last column is the result/target column

    
    col = torch.argmax(chosen_features[i],dim=0)
    
    columnValues = [row[col] for row in rows]

    #unique values
    lsUnique = list(set(columnValues))

    for value in lsUnique:
        (set1, set2) = divideSet(rows, col, value)

        # Gain -- Entropy or Gini
        p = float(len(set1)) / len(rows)
        gain = currentScore - p*evaluationFunction(set1) - (1-p)*evaluationFunction(set2)
        bestGain = gain
        bestAttribute = (col, value)
        bestSets = (set1, set2)

    
    if bestGain > 0:
        trueBranch = growDecisionTreeFrom(bestSets[0],2*i+1, chosen_features, evaluationFunction)
        falseBranch = growDecisionTreeFrom(bestSets[1],2*i+2, chosen_features, evaluationFunction)
        return DecisionTree(col=bestAttribute[0], index=i, value=bestAttribute[1], trueBranch=trueBranch,
                            falseBranch=falseBranch, summary=dcY)
    else:
        return DecisionTree(results=uniqueCounts(rows), summary=dcY)



def classify(observations, tree, feature_list, dataMissing=False):
    """Classifies the observationss according to the tree.
    dataMissing: true or false if data are missing or not. """

    def classifyWithoutMissingData(observations, tree, feature_list, logP):
        if tree.results != None:  # leaf
            return tree.results, logP
        else:
            v = observations[tree.col]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value: branch = tree.trueBranch
                else: branch = tree.falseBranch
            else:
                if v == tree.value: branch = tree.trueBranch
                else: branch = tree.falseBranch

        logP *= torch.max(feature_list[tree.index])
        
        return classifyWithoutMissingData(observations, branch, feature_list, logP)


    def classifyWithMissingData(observations, tree, feature_list, logP):
        if tree.results != None:  # leaf
            return tree.results, node_list
        else:
            v = observations[tree.col]
            if v == None:
                tr = classifyWithMissingData(observations, tree.trueBranch)
                fr = classifyWithMissingData(observations, tree.falseBranch)
                tcount = sum(tr.values())
                fcount = sum(fr.values())
                tw = float(tcount)/(tcount + fcount)
                fw = float(fcount)/(tcount + fcount)
                result = defaultdict(int) # Problem description: http://blog.ludovf.net/python-collections-defaultdict/
                for k, v in tr.items(): result[k] += v*tw
                for k, v in fr.items(): result[k] += v*fw
                return dict(result)
            else:
                branch = None
                if isinstance(v, int) or isinstance(v, float):
                    if v >= tree.value: branch = tree.trueBranch
                    else: branch = tree.falseBranch
                else:
                    if v == tree.value: branch = tree.trueBranch
                    else: branch = tree.falseBranch

            logP *= torch.max(feature_list[tree.index]) 
            if branch == tree.trueBranch:
                i = 2**i
            else:
                i = 2**i + 1
            
            return classifyWithMissingData(observations, branch, feature_list, logP)

    # function body
    if dataMissing:
        return classifyWithMissingData(observations, tree, feature_list, 1)
    else:
        return classifyWithoutMissingData(observations, tree, feature_list, 1)


def rl_loss(batch, tree,feature_list, r):
    loss = 0
    acc = 0
    for i in range(len(batch)):
        pred, logP = classify(batch[i], tree, feature_list)
        if batch[i][-1] == max(pred, key=pred.get):
            loss += logP * (1-r)
            acc += 1
        else:
            loss += logP*(-r)
    return -loss, acc/len(batch)

