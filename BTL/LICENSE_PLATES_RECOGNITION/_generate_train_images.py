import cv2
import os 
import numpy as np
import csv

# liệt kê các tập tin ảnh chụp từ thư mục 'raw_images'
path = 'raw_images'
list_file = os.listdir(path)

# file index dùng để tạo tên file khi lưu từng ảnh ký tự
file_index = 0

# bắt đầu duyệt từng tập tin
for file in list_file:
    # đọc ảnh xám
    image_raw = cv2.imread(path + '/' + file, cv2.IMREAD_GRAYSCALE)
    # làm mờ ảnh để lọc nhiễu
    image_raw = cv2.medianBlur(image_raw,19)
    # lấy giá trị độ sáng trung bình của ảnh.
    # mục đích để nhị phân hóa ở bước tiếp theo
    mean_raw = cv2.mean(image_raw)[0]
    # nhị phân hóa
    _, image_bin = cv2.threshold(image_raw, int(mean_raw)+50, 255, cv2.THRESH_BINARY)
    # phát hiện các đường biên trong ảnh.
    # mục đích phát hiện các đường biên để dự đoán khu vực có khả năng chứa một đối tượng nào đó
    contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # bắt đầu duyệt toàn bộ các đường biên tìm được
    for contour in contours:
        # xác định một vùng không gian hình chữ nhật chứa đường biên.
        box = cv2.boundingRect(contour)
        # sau đó trích xuất ra từng thông số của khu vực đó bao gồm x,y, width, height
        x = box[0]
        y = box[1]
        width = box[2]
        height = box[3]

        # ở bước tiếp theo này, ta kiểm tra tính hợp lệ của đường biên dựa vào các thông số.
        # mục đích ta sẽ lọc đi các vùng có khả năng lớn không phải là biển số
        if width < height :
            continue
        if width < 500 or width > 1000: 
            continue
        scale = float(width) / float(height)
        if scale < 1.1 or scale > 2:
            continue

        
        # Sau bước lọc, ta cắt ảnh trong khu vực nhận diện từ ảnh xám
        image_plate = image_raw[y:y+height, x:x+width]
        # thực hiện quá trình nhị phân hóa. 
        image_plate_mean = cv2.mean(image_plate)[0]
        _, image_plate_bin = cv2.threshold(image_plate, image_plate_mean,255, cv2.THRESH_BINARY_INV)

        # tiếp tục thực hiện phát hiện đường biên trong ảnh biển số
        plate_contours, _ = cv2.findContours(image_plate_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for plate_contour in plate_contours:
            character_box = cv2.boundingRect(plate_contour)
            character_x = character_box[0]
            character_y = character_box[1]
            character_width = character_box[2]
            character_height = character_box[3]
            character_size_raito = character_height / character_width
            # lọc sơ bộ vùng chứa đường biên
            if character_size_raito < 1.5 or character_size_raito > 5:
                continue
            if character_width < 50 or character_width > 200:
                continue
            if(character_x < 5 or character_y < 5):
                continue

            # thực hiện cắt ảnh nghi ngờ là chữ số từ ảnh biển số.
            # và thực hiện nhị phân hóa
            image_character = image_plate[character_y:character_y+character_height, character_x:character_x+character_width]
            image_character_mean = cv2.mean(image_character)[0]
            _, image_character_bin = cv2.threshold(image_character, image_character_mean, 255, cv2.THRESH_BINARY_INV)
            cv2.rectangle(image_character_bin, (0,0),(character_width, character_height),(0),2)
            # và lưu nó vào tập tin
            file_index += 1
            cv2.imwrite('character_images/' + str(file_index) + '.bmp', image_character_bin)
            pass
        pass

    # cv2.imshow('image', image_bin)
    # cv2.waitKey(100)
    pass
print('Finished !')