import rnn
import torch
from dtree import *
from loader import *
from plot_tree import *
from gif import *

input_size = 4
hidden_size = 10
num_layers = 10
epochs = 100
r = 0.5
level = 5
learning_rate = 0.00001
def main():
    
    model = rnn.RNN(input_size,hidden_size,num_layers)

    
    init_x = torch.zeros((input_size),requires_grad=True)
    init_h = torch.zeros((hidden_size),requires_grad=True)
    optimizer = torch.optim.Adam([
                                {'params':model.parameters()},
                                {'params':init_x},
                                {'params':init_h}
                                    ], lr=learning_rate)
    bHeader = True
    # the bigger example
    dcHeadings, trainingData = loadCSV('fishiris.csv',bHeader) # demo data from matlab
    print(len(trainingData))
    testData = trainingData[-20:]
    trainingData = trainingData[0:-20]
    for epoch in range(epochs):
        x = init_x
        h = init_h
        pred = []
        pred.append(x)
        for i in range(pow(2,level)-1):
            h, x = model(h, x)
            pred.append(x)
        
        
        decisionTree = growDecisionTreeFrom(trainingData, 0, pred, evaluationFunction=gini)
        optimizer.zero_grad()
        loss, acc = rl_loss(trainingData, decisionTree, pred, r)
        
        loss.backward(retain_graph=True)
        optimizer.step()
        test_loss, test_acc = rl_loss(testData, decisionTree,pred, r)
        print(str(test_loss.item()) + " "+ str(test_acc))
        #prune(decisionTree, 0.8, notify=True) # notify, when a branch is pruned (one time in this example)
        # result = plot(decisionTree,dcHeadings)
        # #print(result)
        dot_data = dotgraph(decisionTree,dcHeadings)
        graph = pydotplus.graph_from_dot_data(dot_data)
        print(epoch)
        graph.write_png("./tree_images/iris_"+str(epoch)+".png")
    gif(epochs)

if __name__ == '__main__':
    main()