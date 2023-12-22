import cv2
import os 
import numpy as np


def save_number_gray(index, image):
    output_path = 'character_images/grays/' + str(file_index) + '.bmp'
    cv2.imwrite(output_path, image)
    pass


def save_number_blackwhite(index, image):
    output_path = 'character_images/blackwhites/' + str(file_index) + '.bmp'
    cv2.imwrite(output_path, image)
    pass


path = 'plate_images'
list_files = os.listdir(path)
file_index = 0

for file_name in list_files:
    file_path = path + '/' + file_name
    mat = cv2.imread(file_path)
    mat_gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    mat_blurred = cv2.GaussianBlur(mat_gray, (17, 17), 0)
    mean = cv2.mean(mat_blurred)[0]
    thresh, mat_bin = cv2.threshold(mat_blurred, int(mean), 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(mat_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        box = cv2.boundingRect(contour)
        x = box[0]
        y = box[1]
        width = box[2]
        height = box[3]
        if width > height :
            continue
        if width < 50 or width > 400: 
            continue
        
        mat_crop_gray = mat_blurred[y:y+height, x:x+width]
        cv2.rectangle(mat_crop_gray, (0,0, width, height), (0), 2)
        save_number_gray(file_index, mat_crop_gray)


        mat_crop_bin = mat_bin[y:y+height, x:x+width]
        cv2.rectangle(mat_crop_bin, (0,0, width, height), (0), 2)
        erode_thresh = width / 10
        kernel = np.ones((int(erode_thresh), int(erode_thresh)), np.uint8) 
        mat_crop_bin = cv2.erode(mat_crop_bin, kernel)  
        save_number_blackwhite(file_index, mat_crop_bin)

        file_index+=1

print("Finished !")