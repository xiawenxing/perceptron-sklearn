import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from perception.plot import plot
from perception.todo import func as perception_func
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import Imputer
# some usable model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier

import warnings
warnings.filterwarnings('ignore')
#os.chdir('/lqq/FinMKT/venv')
prety = ['sklearn','Perception']

def data_preprocess(data):
    # your code here
    # example:
    # 缺失值填充(众数填充)、onehot编码
    x = pd.get_dummies(data)
    imp = Imputer(missing_values='NaN', strategy='most_frequent',axis=0)
    df = pd.DataFrame(x)
    imp.fit(df)
    df = imp.transform(x)
    i = 0    
    for key in x:
        x.loc[:,key] = df[:,i]
        i+=1
    # your code end
    return x

def predict(x_train, x_test, y_train, pretype):
    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'
    # your code here end
    xtrain = np.squeeze(np.matrix(pd.DataFrame(x_train))).T
    ytrain = np.asmatrix(np.asarray(np.squeeze(pd.DataFrame(y_train))))
    ytrain[ytrain=="yes"]=1
    ytrain[ytrain=="no"]=-1
    xtest = np.squeeze(np.matrix(pd.DataFrame(x_test)))
    y_pred = [0]*len(xtest)

    if pretype=="sklearn":
        model = KNeighborsClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
    elif pretype=="Perception":
        w_g = perception_func(xtrain, ytrain)
        test_N,test_P = xtest.shape
        for i in range(test_N):
            if(float(np.dot(xtest[i,:],w_g[0:test_P,:])+w_g[test_P,0])<=0):
                y_pred[i]='no'
            else:
                y_pred[i]='Yes'
        y_pred = np.asarray(y_pred)

    return y_pred

def split_data(data):
    y = data.y
    x = data.loc[:, data.columns != 'y']
    x = data_preprocess(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    return x_train, x_test, y_train, y_test

def print_result(y_test, y_pred):
    report = confusion_matrix(y_test, y_pred)
    precision = report[1][1] / (report[:, 1].sum())
    recall = report[1][1] / (report[1].sum())
    print('model precision:' + str(precision)[:4] + ' recall:' + str(recall)[:4])

if __name__ == '__main__':
    data = pd.read_csv('bank-additional-full.csv', sep=';')
    x_train, x_test, y_train, y_test = split_data(data)
    for pretype in prety:
        print("predict type:%s---------------", pretype)
        y_pred = predict(x_train, x_test, y_train, pretype)
        print_result(y_test, y_pred)

