import cv2
import os
import pandas as pd
import numpy as np
import ImageHandler
import ImageExtractor

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle




def load_labels():
    objects = {}
    csv_objects = pd.read_csv(os.getcwd() + '/model_traineds/objects.csv').values
    cnt = csv_objects.shape[0]
    for i in range(cnt):
        id = csv_objects[i][0]
        name = csv_objects[i][1]
        objects[id] = name
    return objects

def load_models():
    lnr_model_reader = open(os.getcwd() + '/model_traineds/lnr.model', 'rb')
    lnr = pickle.load(lnr_model_reader)
    lnr_model_reader.close()

    knn_model_reader = open(os.getcwd() + '/model_traineds/knn.model', 'rb')
    knn = pickle.load(knn_model_reader)
    knn_model_reader.close()

    lgr_model_reader = open(os.getcwd() + '/model_traineds/lgr.model', 'rb')
    lgr = pickle.load(lgr_model_reader)
    lgr_model_reader.close()

    svm_model_reader = open(os.getcwd() + '/model_traineds/lgr.model', 'rb')
    svm = pickle.load(svm_model_reader)
    svm_model_reader.close()
    
    return lnr, knn, lgr, svm

def detect_with_lnr(lnr, image):
    info = ImageExtractor.image_extract_info(image)
    test_data = pd.DataFrame(info).T
    id = lnr.predict(test_data)[0]
    name = '?'
    if id in objects:
        name = objects[id]
    return name

def detect_with_knn(knn, image):
    info = ImageExtractor.image_extract_info(image)
    test_data = pd.DataFrame(info).T
    id = knn.predict(test_data)[0]
    name = '?'
    if id in objects:
        name = objects[id]
    return name

def detect_with_lgr(lgr, image):
    info = ImageExtractor.image_extract_info(image)
    test_data = pd.DataFrame(info).T
    id = lgr.predict(test_data)[0]
    name = '?'
    if id in objects:
        name = objects[id]
    return name

def detect_with_svm(svm, image):
    info = ImageExtractor.image_extract_info(image)
    test_data = pd.DataFrame(info).T
    id = svm.predict(test_data)[0]
    name = '?'
    if id in objects:
        name = objects[id]
    return name



if __name__ == '__main__':
    objects = load_labels()
    lnr, knn, lgr, svm = load_models()

    path = 'test_images'
    list_files = os.listdir(path)
    for file_name in list_files:
        file_path = path + '/' + file_name
        image_draw = None
        image_raw = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image_raw = cv2.medianBlur(image_raw, 19)
        image_draw = cv2.cvtColor(image_raw, cv2.COLOR_GRAY2BGR)
        mean = cv2.mean(image_raw)[0]
        thresh, image_bin = cv2.threshold(image_raw, int(mean), 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            box = cv2.boundingRect(contour)
            x = box[0]
            y = box[1]
            width = box[2]
            height = box[3]

            if width > height :
                continue

            if width < 40 or width > 400: 
                continue

            if height < 200 or height > 500: 
                continue
            
            mat_bin_crop = image_bin[y:y+height, x:x+width]

            white_point = cv2.countNonZero(mat_bin_crop)
            fill = white_point * 1.0 / (height * width)
            if fill < 0.2:
                continue

            name = detect_with_knn(knn, mat_bin_crop)
    
            # draw
            cv2.rectangle(image_draw, box, (255,0,0),2)
            cv2.putText(image_draw, str(name), (box[0]+5, box[1]+50), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255),3)

        cv2.imshow('detection', image_draw)
        cv2.waitKey(1000)
    pass