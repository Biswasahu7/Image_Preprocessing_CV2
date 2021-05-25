# ********************************
# Preprocessing Images for OCR
# ********************************

# Importing required libraries...
import cv2
import numpy as np
import easyocr
from resizeimage import resizeimage
#
# # DEFINING EASY_OCR with language
reader = easyocr.Reader(['en'])

# img = cv2.imread("/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Crop_Images/image_801.jpg")

img = cv2.imread("/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Crop_Images/image_100.jpg")
# img = cv2.imread("/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Crop_Images/image_713.jpg")


# Original images
result = reader.readtext(img)
print(result)
print("Original Image result-{}".format(result[0][1]))
cv2.imshow('Original Image', img)
dimensions = img.shape
# print("Image dimension-{}".format(dimensions))

# resizeimg = resizeimage.resize_crop(img, [30, 110,3])
# dimensions = x.shape
# print("Resize image dimension-{}".format(x))

# Inverted images

inverted_image = cv2.bitwise_not(img)
result1 = reader.readtext(inverted_image)
print(result1)
print("Inverted Image result-{}".format(result1[0][1]))
cv2.imshow('inverted Image', inverted_image)


# Gray image

# # Convert Image to  Binarization format
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Image passing to function to convert grayscale
gray_image =  grayscale(img)
result2 = reader.readtext(gray_image)
print(result2)
print("Gray Image result-{}".format(result2[0][1]))
cv2.imshow('Gray Image', gray_image)


# Black & white image

# Manage thersh hold for image (Black & White image)
thersh, im_bw = cv2.threshold(img,120,60, cv2.THRESH_BINARY)

result3 = reader.readtext(im_bw)
print(result3)
print("Black & White Image result-{}".format(result2[0][1]))
cv2.imshow('Black & white', im_bw)

# Noise Removal
def noice_removal(image):
    import numpy as np
    kernal =  np.ones((1,1),np.uint8)
    image = cv2.dilate(image,kernal,iterations=1)
    kernal = np.ones((1,1),np.uint8)
    image = cv2.erode(image,kernal,iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernal)
    image = cv2.medianBlur(image,3)
    return image

imagenr =  noice_removal(inverted_image)

result4 = reader.readtext(imagenr)
print(result4)
# print("Noise remove Image result-{}".format(result4[0][1]))
cv2.imshow('Noise remove image', imagenr)
# print("Noise remove image result-{}".format(result4[0][1]))
# dimensions1 = img.shape
# print("Image dimension-{}".format(dimensions1))

# Wait time
cv2.waitKey(0)
