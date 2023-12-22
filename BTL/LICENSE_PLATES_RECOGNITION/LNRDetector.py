import pandas as pd
import os
import numpy as np
from matplotlib import pyplot
import scipy.optimize as opt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle
import cv2




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
        image = cv2.resize(image, size)
        mean = image.mean()
        _, bin = cv2.threshold(image, mean, 255, cv2.THRESH_BINARY_INV)
        bytes = np.asarray(bin).reshape(-1)
        print(bytes)
        predict_data = pd.DataFrame(bytes).T
        # id = round(self.model.predict(predict_data)[0])
        # if id in self.objects:
        #     return self.objects[id]
        # else:
        #     return '?'
        return '?'