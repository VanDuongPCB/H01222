import cv2
import os 
import numpy as np
import csv
import ImageHandler

path = 'character_images'
list_dir = os.listdir(path)
file_bin = open('train_datas/number_data_bin.csv', 'w')

for dir in list_dir:
    character = dir
    
    list_file = os.listdir(path + '/' + dir)
    for file in list_file:
        file_path = path + '/' + character + '/' + file
        image_raw = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        box_raito, contour_per_box, white_pixel_per_contour, contour_hull_area_per_box, white_pixel_count_1_per_box, white_pixel_count_2_per_box, white_pixel_count_3_per_box, white_pixel_count_4_per_box = ImageHandler.get_data(image_raw)

        data_row_bin = ''
        data_row_bin += str(box_raito) + ','
        data_row_bin += str(contour_per_box) + ','
        data_row_bin += str(white_pixel_per_contour) + ','
        data_row_bin += str(contour_hull_area_per_box) + ','
        data_row_bin += str(white_pixel_count_1_per_box) + ','
        data_row_bin += str(white_pixel_count_2_per_box) + ','
        data_row_bin += str(white_pixel_count_3_per_box) + ','
        data_row_bin += str(white_pixel_count_4_per_box) + ','
        data_row_bin += character + '\n'
        file_bin.write(data_row_bin)

file_bin.close()
print('Finished !')