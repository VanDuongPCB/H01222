import cv2
import os 
import numpy as np
import csv
import ImageHandler
import ImageExtractor

path = 'character_images/blackwhites'
list_dir = os.listdir(path)
file_bin = open('train_datas/train_data.csv', 'w')
for dir in list_dir:
    character = dir
    list_file = os.listdir(path + '/' + dir)
    for file in list_file:
        file_path = path + '/' + character + '/' + file
        image_raw = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        line_data = ImageExtractor.image_extract_info(file_path, image_raw)
        line_data.append(character)
        line_data.append(file_path)
        print(line_data)


        data_row_bin = ''
        for val in line_data:
            data_row_bin = data_row_bin + str(val) + ','

#         data_row_bin += str(line_data[0]) + ','
#         data_row_bin += str(line_data[1]) + ','
#         data_row_bin += str(line_data[2]) + ','
#         data_row_bin += str(line_data[3]) + ','
#         data_row_bin += str(line_data[4]) + ','
#         data_row_bin += str(line_data[5]) + ','
#         data_row_bin += str(line_data[6]) + ','
#         # data_row_bin += str(line_data[7]) + ','
# #         data_row_bin += str(white_pixel_count_3_per_box) + ','
# #         data_row_bin += str(white_pixel_count_4_per_box) + ','
        data_row_bin += '\n'
        file_bin.write(data_row_bin)

file_bin.close()
print('Finished !')