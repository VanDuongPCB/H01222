import pandas as pd
import os
import numpy as np
from matplotlib import pyplot
import scipy.optimize as opt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pickle




class KNNDetector:
    model = None
    objects = {}

    def __init__(self):
        model_file_path = os.getcwd() + '/model_traineds/knn.model'
        file_model_reader = open(model_file_path, 'rb')
        self.model = pickle.load(file_model_reader)

        csv_objects = pd.read_csv(os.getcwd() + '/model_traineds/knn.csv').values
        cnt = csv_objects.shape[0]
        for i in range(cnt):
            id = csv_objects[i][0]
            name = csv_objects[i][1]
            self.objects[id] = name
        pass

    def detect(self, image, size):
        pass