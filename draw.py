import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import graphviz
import sklearn
matplotlib.use('Agg')

def PickleLoad(in_file):
    fid = open(in_file, 'rb')
    data = pickle.load(fid)
    fid.close()
    return data


def simpleShow(data_name, metric):
    result = PickleLoad('result_' + data_name + '_' + metric + '.pkl')
    baseline = PickleLoad('baseline.pkl')
    length = len(result)
    for score in ['acc', 'f1', 'auc']:
        valid_result = np.concatenate([np.array(result[i][0]['valid'][score]).reshape(1, -1) \
            for i in range(length)], axis=0)
        test_result = np.concatenate([np.array(result[i][0]['test'][score]).reshape(1, -1) \
            for i in range(length)], axis=0)
        valid_bl = baseline[data_name]['valid'][score]
        test_bl = baseline[data_name]['test'][score]

        plt.figure()
        plt.errorbar(np.arange(11), valid_result.mean(axis=0), yerr=valid_result.std(axis=0), label='valid')
        plt.errorbar(np.arange(11), test_result.mean(axis=0), yerr=test_result.std(axis=0), label='test')
        plt.plot(np.arange(11), [valid_bl] * 11, label='valid_bl')
        plt.plot(np.arange(11), [test_bl] * 11, label='test_bl')
        plt.legend()
        plt.title("%s  %s" %(data_name, score))
        plt.savefig('./figure2/' + data_name + '_' + score + '.png')
        #plt.show()
        #plt.close()


def sampleCurve(data_name, metric):
    result = PickleLoad('result_' + data_name + '_' + metric + '.pkl')
    baseline = PickleLoad('baseline.pkl')
    for i in range(len(result)):
        plt.figure()
        plt.plot(result[i][1][0], label='train')
        plt.plot(result[i][1][1], label='valid')
        plt.plot(result[i][1][2], label='test')
        plt.legend()
        plt.title("CART: valid: %f, test: %f" \
            %(baseline[data_name]['valid'][metric], baseline[data_name]['test'][metric]))
        plt.savefig('figure/' + data_name + '_' + metric + '_sample_curve.png')
        #plt.show()
        #plt.close()


def allEpoch(data_name, metric):
    result = PickleLoad('result_' + data_name + '_' + metric + '.pkl')
    baseline = PickleLoad('baseline.pkl')
    length = len(result)
    for score in ['acc', 'f1', 'auc']:
        valid_result = np.array([result[i][0]['valid'][score][-1] for i in range(length)])
        test_result = np.array([result[i][0]['test'][score][-1] for i in range(length)])
        valid_bl = baseline[data_name]['valid'][score]
        test_bl = baseline[data_name]['test'][score]

        plt.figure()
        plt.plot(valid_result, label='valid')
        plt.plot(test_result, label='test')
        plt.plot(np.arange(length), [valid_bl] * length, label='valid_bl')
        plt.plot(np.arange(length), [test_bl] * length, label='test_bl')
        plt.legend()
        plt.title("%s  %s" %(data_name, score))
        plt.savefig('./figure2/' + data_name + '_' + score + '.png')
        #plt.show()
        #plt.close()


def dataResult(data_name, metric):
    print (data_name)
    baseline = PickleLoad('baseline.pkl')
    print ('baseline')
    print ('valid: %f    test:%f' %(baseline[data_name]['valid'][metric], baseline[data_name]['test'][metric]))
    result = PickleLoad('result_' + data_name + '_' + metric + '.pkl')
    length = len(result)
    valid = np.array([result[i][0]['valid'][metric][-1] for i in range(length)])
    test = np.array([result[i][0]['test'][metric][-1] for i in range(length)])

    print ('average score')
    print ('valid:', valid.mean(), valid.std())
    print ('test:', test.mean(), test.std())
    print ('best score')
    idx = np.argmax(valid)
    print ('valid:', valid[idx])
    print ('test:', test[idx])


def drawSklearnTree(tree, name):
    dot_data = sklearn.tree.export_graphviz(tree, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render(name)



if __name__ == '__main__':

    for data_name in ['pima', 'heart', 'german', 'breast_cancer', 'credit']:
        for metric in ['auc','acc','f1']:
            print(metric + " :")
            dataResult(data_name, metric)
            #simpleShow(data_name, metric)
            #sampleCurve(data_name, metric)
        print("--------------------------")
    
    # data_name = 'credit'
    # metric = 'f1'
    # dataResult(data_name, metric)
    # #simpleShow(data_name, metric)
    # sampleCurve(data_name, metric)
    