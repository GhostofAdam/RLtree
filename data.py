import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import sklearn
import seaborn as sns
<<<<<<< HEAD
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from draw import drawSklearnTree
=======
>>>>>>> refs/remotes/origin/master

def PickleSave(out_file, data):
    fid = open(out_file, 'wb')
    pickle.dump(data, fid, True)
    fid.close()

def PickleLoad(in_file):
    fid = open(in_file, 'rb')
    data = pickle.load(fid)
    fid.close()
    return data

def parseAdult(file_name, category):

    fr = open(file_name + '.csv', 'r')
    data = fr.readlines()
    fr.close()

    complete_data = []
    for i in range(len(data)):
        attributes = data[i].replace(' ', '').replace('.', '').replace('\n', '').split(',')
        # skip samples with missing value
        if '?' in attributes:
            continue
        complete_data.append(attributes)
    sample_num = len(complete_data)

    age = np.zeros([sample_num, 1])
    workclass = np.zeros([sample_num, 8])
    fnlwgt = np.zeros([sample_num, 1])
    education = np.zeros([sample_num, 16])
    education_num = np.zeros([sample_num, 1])
    marital_status = np.zeros([sample_num, 7])
    occupation = np.zeros([sample_num, 14])
    relationship = np.zeros([sample_num, 6])
    race = np.zeros([sample_num, 5])
    sex = np.zeros([sample_num, 1])
    capital_gain = np.zeros([sample_num, 1])
    capital_loss = np.zeros([sample_num, 1])
    hours_per_week = np.zeros([sample_num, 1])
    native_country = np.zeros([sample_num, 1])
    label = np.zeros(sample_num, dtype='int')

    for i in range(sample_num):
        attributes = complete_data[i]
        age[i, 0] = float(attributes[0])
        for j in range(8):
            if (attributes[1] == category['workclass'][j]):
                workclass[i, j] = 1
                break
        fnlwgt[i, 0] = float(attributes[2])
        for j in range(16):
            if (attributes[3] == category['education'][j]):
                education[i, j] = 1
                break
        education_num[i, 0] = float(attributes[4])
        for j in range(7):
            if (attributes[5] == category['marital-status'][j]):
                marital_status[i, j] = 1
                break
        for j in range(14):
            if (attributes[6] == category['occupation'][j]):
                occupation[i, j] = 1
                break
        for j in range(6):
            if (attributes[7] == category['relationship'][j]):
                relationship[i, j] = 1
                break
        for j in range(5):
            if (attributes[8] == category['race'][j]):
                race[i, j] = 1
                break
        if (attributes[9] == 'Female'):
            sex[i, 0] = 1
        capital_gain[i, 0] = float(attributes[10])
        capital_loss[i, 0] = float(attributes[11])
        hours_per_week[i, 0] = float(attributes[12])
        if (attributes[13] != 'United-States'):
            native_country[i, 0] = 1

        if (attributes[-1] == '<=50K'):
            label[i] = 0
        else:
            label[i] = 1

    features = np.concatenate([age, workclass, fnlwgt, education, education_num, marital_status, \
               occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, \
               native_country], axis=1)

    return features, label


class Adult():
    
    def __init__(self):

        category = {}
        category['workclass'] = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', \
                                 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
        category['education'] = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', \
                                 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', \
                                 '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
        category['marital-status'] = ['Married-civ-spouse', 'Divorced', 'Never-married', \
                                      'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
        category['occupation'] = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', \
                                  'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', \
                                  'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', \
                                  'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
        category['relationship'] = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
        category['race'] = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']

        X_train, Y_train = parseAdult('adult_train', category)
        X_test, Y_test = parseAdult('adult_test', category)
        self.features = np.concatenate([X_train, X_test], axis=0)
        self.label = np.append(Y_train, Y_test)

        order = np.arange(self.features.shape[0])
        np.random.shuffle(order)

        self.features = self.features[order, :]
        self.label = self.label[order]


class Heart():
    
    def __init__(self):

        fr = open('heart.csv', 'r')
        data = fr.readlines()
        fr.close()

        attribute_num = 13
        features = [np.zeros([len(data), 1])] * attribute_num
        features[2] = np.zeros([len(data), 4])
        features[6] = np.zeros([len(data), 3])
        features[12] = np.zeros([len(data), 3])
        label = np.zeros(len(data), dtype='int')

        for i in range(len(data)):
            attributes = data[i].replace('\n', '').split(' ')
            for j in [0, 1, 3, 4, 5, 7, 8, 9, 10, 11]:
                features[j][i, 0] = float(attributes[j])
            features[2][i, int(float(attributes[2])) - 1] = 1
            features[6][i, int(float(attributes[6]))] = 1
            if (float(attributes[12]) == 3):
                features[12][i, 0] = 1
            elif (float(attributes[12]) == 6):
                features[12][i, 1] = 1
            elif (float(attributes[12]) == 7):
                features[12][i, 2] = 1
            label[i] = int(attributes[-1]) - 1
        features = np.concatenate(features, axis=1)

        order = np.arange(features.shape[0])
        np.random.shuffle(order)

        self.features = features[order, :]
        self.label = label[order]


class German():
    
    def __init__(self):

        fr = open('german.csv', 'r')
        data = fr.readlines()
        fr.close()

        feature_num = 24
        features = np.zeros([len(data), feature_num], dtype='float')
        label = np.zeros(len(data), dtype='int')

        for i in range(len(data)):
            attributes = data[i].split()
            for j in range(feature_num):
                features[i, j] = float(attributes[j])
            label[i] = int(attributes[-1]) - 1

        order = np.arange(features.shape[0])
        np.random.shuffle(order)

        self.features = features[order, :]
        self.label = label[order]


class Pima():
    
    def __init__(self):

        fr = open('pima.csv', 'r')
        data = fr.readlines()
        fr.close()

        features = np.zeros([len(data), 8], dtype='float')
        label = np.zeros(len(data), dtype='int')

        for i in range(len(data)):
            attributes = data[i].replace('\n', '').split(',')
            for j in range(8):
                features[i, j] = float(attributes[j])
                label[i] = int(attributes[-1])

        order = np.arange(features.shape[0])
        np.random.shuffle(order)

        self.features = features[order, :]
        self.label = label[order]


class Glass():
    
    def __init__(self):

        fr = open('glass.csv', 'r')
        data = fr.readlines()
        fr.close()

        features = np.zeros([len(data), 9], dtype='float')
        label = np.zeros(len(data), dtype='int')

        for i in range(len(data)):
            attributes = data[i].split(',')
            for j in range(9):
                features[i, j] = float(attributes[j + 1])
            if (int(attributes[-1]) <= 4):
                label[i] = 0
            else:
                label[i] = 1

        order = np.arange(features.shape[0])
        np.random.shuffle(order)

        self.features = features[order, :]
        self.label = label[order]


class BreastCancer():
    
    def __init__(self):

        data = load_breast_cancer()
        features = data['data']
        label = data['target']

        order = np.arange(features.shape[0])
        np.random.shuffle(order)

        self.features = features[order, :]
        self.label = label[order]


class Chess():
    
    def __init__(self):

        fr = open('chess.csv', 'r')
        data = fr.readlines()
        fr.close()

        feature_num = 36
        features = np.zeros([len(data), feature_num], dtype='float')
        label = np.zeros(len(data), dtype='int')

        for i in range(len(data)):
            attributes = data[i].replace('\n', '').split(',')
            for j in range(feature_num):
                if (attributes[j] == 'f'):
                    features[i, j] = 1
                else:
                    features[i, j] = 0
            if (attributes[-1] == 'won'):
                label[i] = 1
            else:
                label[i] = 0

        order = np.arange(features.shape[0])
        np.random.shuffle(order)

        self.features = features[order, :]
        self.label = label[order]


class Spam():
    
    def __init__(self):

        fr = open('spam.csv', 'r')
        data = fr.readlines()
        fr.close()

        feature_num = 57
        features = np.zeros([len(data), feature_num], dtype='float')
        label = np.zeros(len(data), dtype='int')

        for i in range(len(data)):
            attributes = data[i].replace('\n', '').split(',')
            for j in range(feature_num):
                features[i, j] = float(attributes[j])
            label[i] = int(attributes[-1])

        order = np.arange(features.shape[0])
        np.random.shuffle(order)

        self.features = features[order, :]
        self.label = label[order]


class Mammo():
    
    def __init__(self):

        fr = open('mammo.csv', 'r')
        data = fr.readlines()
        fr.close()

        feature_num = 12
        features = np.zeros([len(data), feature_num], dtype='float')
        label = np.zeros(len(data), dtype='int')

        count = 0
        for i in range(len(data)):
            attributes = data[i].replace('\n', '').split(',')
            if '?' in attributes:
                continue
            features[count, 0] = float(attributes[0])
            features[count, 1] = float(attributes[1])
            features[count, 1 + int(attributes[2])] = 1
            features[count, 5 + int(attributes[3])] = 1
            features[count, 11] = float(attributes[4])
            label[count] = int(attributes[-1])
            count += 1

        features = features[:count]
        label = label[:count]

        order = np.arange(features.shape[0])
        np.random.shuffle(order)

        self.features = features[order, :]
        self.label = label[order]


class Australia():
    
    def __init__(self):

        fr = open('australia.csv', 'r')
        data = fr.readlines()
        fr.close()

        num = np.array([1,1,1,3,14,9,1,1,1,1,1,3,1,1])
        feature_num = num.sum()
        num = np.cumsum(num)
        num = np.insert(num, 0, 0)
        features = np.zeros([len(data), feature_num], dtype='float')
        label = np.zeros(len(data), dtype='int')

        for i in range(len(data)):
            attributes = data[i].replace('\n', '').split(' ')
            for j in range(14):
                if j in [3,4,5,11]:
                    features[i, num[j] + int(attributes[j]) - 1] = 1
                else:
                    features[i, num[j]] = float(attributes[j])
            label[i] = int(attributes[-1])

        order = np.arange(features.shape[0])
        np.random.shuffle(order)

        self.features = features[order, :]
        self.label = label[order]


class Monk:
    
    def __init__(self, idx):

        fr = open('monk_' + str(idx) + '.csv', 'r')
        data = fr.readlines()
        fr.close()

        num = np.array([3,3,2,3,4,2])
        feature_num = num.sum()
        num = np.cumsum(num)
        num = np.insert(num, 0, 0)
        features = np.zeros([len(data), feature_num], dtype='float')
        label = np.zeros(len(data), dtype='int')

        for i in range(len(data)):
            attributes = data[i].replace('\n', '').split(' ')
            for j in range(6):
                features[i, num[j] + int(attributes[j + 1]) - 1] = 1
            label[i] = int(attributes[0])

        order = np.arange(features.shape[0])
        np.random.shuffle(order)

        self.features = features[order, :]
        self.label = label[order]


class Bernoulli:

    def __init__(self, size=500, num=4):
        #interval = int(num / 4)
        self.features = np.zeros([size, num], dtype='float')
        self.label = np.zeros(size, dtype='int')
        for i in range(size):
            self.features[i] = np.random.binomial(1, 0.5, num)
            #for j in range(4):
            #    self.features[i, num + j] = self.features[i, j*interval:(j+1)*interval].sum()
            self.label[i] = (self.features[i].sum() >= (num / 2.0))

<<<<<<< HEAD
class Credit():

    def __init__(self):
        
        df = pd.read_csv("./application_train.csv")
        # le = LabelEncoder()
        df = df.dropna(axis=1,how='any',thresh=0.9*len(df))
        df = df.sample(frac=1)
        str_dic = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY',
        'NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE',
        'FONDKAPREMONT_MODE','HOUSETYPE_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE']
        for s in str_dic:
            if s in df.keys():
                df[s] = pd.get_dummies(df[s], prefix=s)
            # df[s] = le.fit_transform(df[s])
        for index, row in df.iteritems():
            if pd.isna(df[index]).all():
                df[index] = df[index].fillna(axis=0,method='ffill',value=0)
        df = df.fillna(0)
        
        self.label = df['TARGET'].values
        df = df.drop(columns = ['TARGET'])
        self.features = df.values
        


def load(data, train_percent, valid_percent, noised = False, frac = 0.1):
=======


def load(data, train_percent, valid_percent):
>>>>>>> refs/remotes/origin/master
    train_num = int(data.features.shape[0] * train_percent)
    valid_num = int(data.features.shape[0] * (train_percent + valid_percent))
    X_train = data.features[:train_num, :]
    X_valid = data.features[train_num:valid_num, :]
    X_test = data.features[valid_num:, :]
    Y_train = data.label[:train_num]
    Y_valid = data.label[train_num:valid_num]
    Y_test = data.label[valid_num:]
    data = {}
    data['X_train'] = X_train
<<<<<<< HEAD
    
    if noised:
        data['Y_train'] = label_noise(Y_train,frac)
    else:
        data['Y_train'] = Y_train

=======
    data['Y_train'] = Y_train
>>>>>>> refs/remotes/origin/master
    data['X_valid'] = X_valid
    data['Y_valid'] = Y_valid
    data['X_test'] = X_test
    data['Y_test'] = Y_test
    #data['split'] = [[0.5]] * X_train.shape[1]
    data['split'] = []
    for i in range(X_train.shape[1]):
<<<<<<< HEAD
        if (set(X_train[:100, i]) == {0, 1} or set(X_train[:100, i]) == {0} or set(X_train[:100, i]) == {1}):
=======
        if (set(X_train[:, i]) == {0, 1} or set(X_train[:, i]) == {0} or set(X_train[:, i]) == {1}):
>>>>>>> refs/remotes/origin/master
            data['split'].append([0.5])
        else:
            data['split'].append([])
    return data

<<<<<<< HEAD
def label_noise(y_data,frac=0.1):
    index = np.random.choice(y_data.shape[0],int(y_data.shape[0]*frac),replace=False)
    for i in index:
        y_data[i] = 1 - y_data[i]
    return y_data
=======
>>>>>>> refs/remotes/origin/master

def baselineRF(data_name):
    data = PickleLoad(data_name + '.pkl')
    tree = RandomForestClassifier(n_estimators=20, criterion='gini', max_features=None, \
        max_depth=5, min_samples_leaf=10, random_state=0)
    tree.fit(data['X_train'], data['Y_train'])
    prediction = tree.predict(data['X_test'])
    print ((prediction == data['Y_test']).sum() / float(prediction.shape[0]))


def basicInfo(data_name):
    data = PickleLoad(data_name + '.pkl')
    train = data['Y_train']
    val = data['Y_valid']
    test = data['Y_test']
    print ('train: ', (train == 1).sum() / float(train.shape[0]), train.shape[0])
    print ('val: ', (val == 1).sum() / float(val.shape[0]), val.shape[0])
    print ('test: ', (test == 1).sum() / float(test.shape[0]), test.shape[0])
<<<<<<< HEAD
    #print ('split: ', data['split'])


def completeBaseline(data_name, max_depth, min_samples_leaf, noised = False):
    if noised:
        data = PickleLoad(data_name + '_noised.pkl')
    else:
        data = PickleLoad(data_name + '.pkl')
    tree = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0)
    tree.fit(data['X_train'], data['Y_train'])
    drawSklearnTree(tree, data_name)
=======
    print ('split: ', data['split'])


def completeBaseline(data_name, max_depth, min_samples_leaf):
    data = PickleLoad(data_name + '.pkl')
    tree = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0)
    tree.fit(data['X_train'], data['Y_train'])
>>>>>>> refs/remotes/origin/master

    result = {'valid':{}, 'test':{}}

    for set_type in ['valid', 'test']:
        pred = tree.predict(data['X_' + set_type])
        pred_proba = tree.predict_proba(data['X_' + set_type])[:, 1]
        result[set_type]['acc'] = sklearn.metrics.accuracy_score(data['Y_' + set_type], pred)
        result[set_type]['f1'] = sklearn.metrics.f1_score(data['Y_' + set_type], pred, average='weighted')
        result[set_type]['auc'] = sklearn.metrics.roc_auc_score(data['Y_' + set_type], pred_proba, average='weighted')
<<<<<<< HEAD
    print(result)
    return result


def allBaseline(noised=False):
=======
    return result


def allBaseline():
>>>>>>> refs/remotes/origin/master
    baseline = {}
    baseline['pima'] = completeBaseline('pima', 4, 10)
    baseline['heart'] = completeBaseline('heart', 4, 5)
    baseline['german'] = completeBaseline('german', 5, 5)
    baseline['breast_cancer'] = completeBaseline('breast_cancer', 5, 5)
    #baseline['chess'] = completeBaseline('chess', 5, 5)
    #baseline['adult'] = completeBaseline('adult', 5, 5)
    #baseline['spam'] = completeBaseline('spam', 5, 5)
<<<<<<< HEAD
    #baseline['mammo'] = completeBaseline('mammo', 5, 10)
    #baseline['australia'] = completeBaseline('australia', 4, 10)
    #baseline['monk_1'] = completeBaseline('monk_1', 3, 5)
    #baseline['bernoulli'] = completeBaseline('bernoulli', 4, 1)
    baseline['credit'] = completeBaseline('credit', 5, 5, noised)
    PickleSave('baseline.pkl', baseline)


def generateData(data_name, train_percent, valid_percent, noised = False, frac = 0.1):
=======
    baseline['mammo'] = completeBaseline('mammo', 5, 10)
    baseline['australia'] = completeBaseline('australia', 4, 10)
    baseline['monk_1'] = completeBaseline('monk_1', 3, 5)
    baseline['bernoulli'] = completeBaseline('bernoulli', 4, 1)
    PickleSave('baseline.pkl', baseline)


def generateData(data_name, train_percent, valid_percent):
>>>>>>> refs/remotes/origin/master
    if (data_name == 'adult'):
        data = Adult()
    elif (data_name == 'glass'):
        data = Glass()
    elif (data_name == 'breast_cancer'):
        data = BreastCancer()
    elif (data_name == 'pima'):
        data = Pima()
    elif (data_name == 'heart'):
        data = Heart()
    elif (data_name == 'german'):
        data = German()
    elif (data_name == 'chess'):
        data = Chess()
    elif (data_name == 'spam'):
        data = Spam()
    elif (data_name == 'mammo'):
        data = Mammo()
    elif (data_name == 'australia'):
        data = Australia()
    elif (data_name == 'monk_1'):
        data = Monk(1)
    elif (data_name == 'bernoulli'):
        data = Bernoulli(num=8)
<<<<<<< HEAD
    elif (data_name == 'credit'):
        data = Credit()
    if noised:
        PickleSave(data_name + '_noised.pkl', load(data, train_percent, valid_percent, noised, frac))
    else:
        PickleSave(data_name + '.pkl', load(data, train_percent, valid_percent, noised))


def baselineDT(data_name, noised=False):
=======
    PickleSave(data_name + '.pkl', load(data, train_percent, valid_percent))


def baselineDT(data_name):
>>>>>>> refs/remotes/origin/master
    data = PickleLoad(data_name + '.pkl')
    print (data['X_train'].shape[1])
    tree = DecisionTreeClassifier(criterion='gini', max_depth=8, min_samples_leaf=1, random_state=0)
    tree.fit(data['X_train'], data['Y_train'])

    print ('auc')
    r1 = tree.predict_proba(data['X_valid'])[:, 1]
    r2 = tree.predict_proba(data['X_test'])[:, 1]
    print (sklearn.metrics.roc_auc_score(data['Y_valid'], r1, average='weighted'), sklearn.metrics.roc_auc_score(data['Y_test'], r2, average='weighted'))

    print ('accuracy')
    r1 = tree.predict(data['X_valid'])
    r2 = tree.predict(data['X_test'])
    print (sklearn.metrics.accuracy_score(data['Y_valid'], r1), sklearn.metrics.accuracy_score(data['Y_test'], r2))
    
    print ('f1')
    print (sklearn.metrics.f1_score(data['Y_valid'], r1, average='weighted'), sklearn.metrics.f1_score(data['Y_test'], r2, average='weighted'))
    


if __name__ == '__main__':
<<<<<<< HEAD
    # data = Credit()
    # print(data.features)
    # print(data.label)
    generateData('credit',0.8, 0.1)
    allBaseline(False)

    data_name = 'credit'
=======
    allBaseline()

    data_name = 'german'
>>>>>>> refs/remotes/origin/master
    print (data_name)
    basicInfo(data_name)