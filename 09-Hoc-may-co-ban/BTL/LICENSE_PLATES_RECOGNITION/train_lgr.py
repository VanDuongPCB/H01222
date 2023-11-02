import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle


duongDan = os.getcwd() + '/train_datas/number_data_bin.csv'
duLieu = pd.read_csv(duongDan)
duLieu


maTran = duLieu.values
rows, cols = maTran.shape
x_train = maTran[:,0:cols-1]

labels = maTran[:, cols-1:cols]
labels = labels.astype('str')
le = LabelEncoder()
y_train = le.fit_transform(labels)

lgr = LogisticRegression()
lgr.fit(x_train, y_train)

model_file_path = os.getcwd() + '/model_traineds/lgr.model'
file_model_writer = open(model_file_path, 'wb')
pickle.dump(lgr, file_model_writer)
file_model_writer.close()

cnt = y_train.shape[0]
object_names = {}
for i in range(cnt):
    name = labels[i]
    id = y_train[i]
    key = tuple(name)
    object_names[key] = id
file_object_writer = open('model_traineds/lgr.csv', 'w')
data_row = ''
for name in object_names:
    id = object_names[name]
    if data_row != '':
        data_row = data_row + '\n'
    data_row += str(id) + ',' + name[0] + ','
file_object_writer.write(data_row)
file_object_writer.close()