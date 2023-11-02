import cv2

def get_data(image):
    box_raito = 0
    contour_per_box = 0
    white_pixel_per_contour = 0
    contour_hull_area_per_box = 0
    white_pixel_count_1_per_box = 0
    white_pixel_count_2_per_box = 0
    white_pixel_count_3_per_box = 0
    white_pixel_count_4_per_box = 0


    image_width = image.shape[0]
    image_height = image.shape[1]
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) < 1 :
        return box_raito,contour_per_box,white_pixel_per_contour,contour_hull_area_per_box,white_pixel_count_1_per_box,white_pixel_count_2_per_box,white_pixel_count_3_per_box,white_pixel_count_4_per_box
    max_contour = contours[0]
    max_contour_index = 0
    for i in range(len(contours)):
        if cv2.contourArea(max_contour) < cv2.contourArea(contours[i]):
            max_contour = contours[i]
            max_contour_index = i

    box = cv2.minAreaRect(max_contour)
    box_width = box[1][1]
    box_height = box[1][0]
    if box_width > box_height :
        temp = box_width
        box_width = box_height
        box_height = temp
        
    box_area = box_width * box_height
    box_raito = box_height / box_width
        
    contour_area = cv2.contourArea(max_contour)
    contour_per_box = contour_area / box_area

    _, mask = cv2.threshold(image, 255, 0, cv2.THRESH_BINARY)
    cv2.drawContours(mask, contours, max_contour_index, (255), cv2.FILLED)
    image = image & mask
    white_pixel_count = cv2.countNonZero(image)
    white_pixel_per_contour = white_pixel_count / contour_area
    if white_pixel_per_contour > 1:
        white_pixel_per_contour = 1

    contour_hull = cv2.convexHull(max_contour)
    contour_hull_area = cv2.contourArea(contour_hull)
    contour_hull_area_per_box = contour_hull_area / box_area
    if contour_hull_area_per_box > 1:
        contour_hull_area_per_box = 1


    box_cx = int(box[0][0])
    box_cy = int(box[0][1])

    mat_crop_1 = image[0:box_cy, 0:box_cx]
    mat_crop_2 = image[0:box_cy, box_cx : image_width]
    mat_crop_3 = image[box_cy : image_height, 0:box_cx]
    mat_crop_4 = image[box_cy : image_height, box_cx : image_width]

    white_pixel_count_1_per_box = cv2.countNonZero(mat_crop_1) / float(box_area)
    white_pixel_count_2_per_box = cv2.countNonZero(mat_crop_2) / float(box_area)
    white_pixel_count_3_per_box = cv2.countNonZero(mat_crop_3) / float(box_area)
    white_pixel_count_4_per_box = cv2.countNonZero(mat_crop_4) / float(box_area)

    return box_raito,contour_per_box,white_pixel_per_contour,contour_hull_area_per_box,white_pixel_count_1_per_box,white_pixel_count_2_per_box,white_pixel_count_3_per_box,white_pixel_count_4_per_box