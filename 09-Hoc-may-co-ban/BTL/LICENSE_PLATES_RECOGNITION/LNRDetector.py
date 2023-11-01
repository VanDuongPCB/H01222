import pandas as pd
import os
import numpy as np
from matplotlib import pyplot
import scipy.optimize as opt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle




class LNRDetector:
    model = None
    objects = {}

    def __init__(self):
        model_file_path = os.getcwd() + '/model_traineds/lnr.model'
        file_model_reader = open(model_file_path, 'rb')
        self.model = pickle.load(file_model_reader)
        self.model.intercept_
        self.model.coef_

        csv_objects = pd.read_csv(os.getcwd() + '/model_traineds/lnr.csv').values
        cnt = csv_objects.shape[0]
        for i in range(cnt):
            id = csv_objects[i][0]
            name = csv_objects[i][1]
            self.objects[id] = name
        pass

    def detect(self, image, size):
        # index = 250
        # x_test = duLieu[index:index+1,0:cols-1]
        # result = linear_regression.predict(x_test)
        # round(result[0])
        pass