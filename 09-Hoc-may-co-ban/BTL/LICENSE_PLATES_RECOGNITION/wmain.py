import cv2
import os
import pandas as pd
import numpy as np
import ImageHandler

from LNRDetector import LNRDetector
from KNNDetector import KNNDetector
from LGRDetector import LGRDetector
from SVMDetector import SVMDetector

lnr = LNRDetector()
knn = KNNDetector()
lgr = LGRDetector()
svm = SVMDetector()

path = 'plate_images'
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
        info = ImageHandler.get_data(mat_bin_crop)
        test_data = pd.DataFrame(info).T
        restult = lnr.model.predict(test_data)
        print(restult)
        

        cv2.imshow('character', mat_bin_crop)
        cv2.waitKey(1000)
        
        # draw
        # cv2.rectangle(image_draw, box, (255,0,0),2)
        # cv2.putText(image_draw, str(id), (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,0),3)

    cv2.imshow('detection', image_draw)
    cv2.waitKey(1000)