import cv2
import os 
import numpy as np
import csv
import ImageHandler

path = 'character_images/blackwhites'
list_dir = os.listdir(path)
for dir in list_dir:
    list_file = os.listdir(path + '/' + dir)
    print(dir, ',', len(list_file))