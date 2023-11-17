import cv2
import numpy as np
import pandas as pd


def image_extract_info(image):
    info = []
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_count = len(contours)
    
    contour_max_area = 0
    contour_max_index = 0
    for index in range(0, contour_count):
        contour_area = cv2.contourArea(contours[index])
        if contour_area > contour_max_area:
            contour_max_area = contour_area
            contour_max_index = index
    
    bounding_box = cv2.boundingRect(contours[contour_max_index])
    image_crop = image[bounding_box[1]:bounding_box[1] + bounding_box[3],bounding_box[0]:bounding_box[2] + bounding_box[0]]
    image_resized = cv2.resize(image_crop, (32,32))
    data = np.array(image_resized ,dtype=np.uint8)
    
    vdata = []
    hdata = []
    for i in range(0,32):
        vdata.append(0)
        hdata.append(0)

    for x in range(0,32):
        for y in range(0,32):
            value = data[y][x]
            vdata[x] += value
            hdata[y] += value

    minval = min(vdata)
    for i in range(0,32):
        vdata[i] = vdata[i] - minval
        info.append(vdata[i])
    
    minval = min(hdata)
    for i in range(0,32):
        hdata[i] = hdata[i] - minval
        info.append(hdata[i])

    return info
    pass