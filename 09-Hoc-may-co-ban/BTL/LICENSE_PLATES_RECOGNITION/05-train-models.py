import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle




def save_labels(labels, y_train):
    cnt = y_train.shape[0]
    object_names = {}
    for i in range(cnt):
        name = labels[i]
        id = y_train[i]
        key = tuple(name)
        object_names[key] = id
    file_object_writer = open('model_traineds/objects.csv', 'w')
    data_row = ''
    for name in object_names:
        id = object_names[name]
        if data_row != '':
            data_row = data_row + '\n'
        data_row += str(id) + ',' + name[0] + ','
    file_object_writer.write(data_row)
    file_object_writer.close()
    pass

def train_linear_regression(x_train, y_train):
    lnr = LinearRegression()
    lnr.fit(x_train, y_train)
    model_file_path = os.getcwd() + '/model_traineds/lnr.model'
    file_model_writer = open(model_file_path, 'wb')
    pickle.dump(lnr, file_model_writer)
    file_model_writer.close()
    pass


def train_logistic_regression(x_train, y_train):
    lgr = LogisticRegression()
    lgr.fit(x_train, y_train)
    model_file_path = os.getcwd() + '/model_traineds/lgr.model'
    file_model_writer = open(model_file_path, 'wb')
    pickle.dump(lgr, file_model_writer)
    file_model_writer.close()
    pass


def train_knn(x_train, y_train):
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    model_file_path = os.getcwd() + '/model_traineds/knn.model'
    file_model_writer = open(model_file_path, 'wb')
    pickle.dump(knn, file_model_writer)
    file_model_writer.close()
    pass

def train_svm(x_train, y_train):
    svm = SVC()
    svm.fit(x_train, y_train)
    model_file_path = os.getcwd() + '/model_traineds/svm.model'
    file_model_writer = open(model_file_path, 'wb')
    pickle.dump(svm, file_model_writer)
    file_model_writer.close()
    pass


if __name__ == '__main__':
    duongDan = os.getcwd() + '/train_datas/train_data.csv'
    duLieu = pd.read_csv(duongDan)
    maTran = duLieu.values
    rows, cols = maTran.shape
    x_train = maTran[:,0:cols-3]
    labels = maTran[:, cols-3:cols-2]
    labels = labels.astype('str')
    le = LabelEncoder()
    y_train = le.fit_transform(labels)

    save_labels(labels, y_train)
    train_linear_regression(x_train, y_train)
    train_logistic_regression(x_train, y_train)
    train_knn(x_train, y_train)
    train_svm(x_train, y_train)
    
    pass
