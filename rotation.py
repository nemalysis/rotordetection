
import os
import cv2 as cv
import pyautogui
import PIL
import matplotlib.pyplot as plt
from pyzbar import pyzbar as zbar
from pprint import pprint as pp
#os.system('xrandr --query')
import time




def get_rotation(image, deg):

    height = image.shape[0]
    width = image.shape[1]

    M = cv.getRotationMatrix2D((width / 2, height / 2), deg, 1)
    image_rotated = cv.warpAffine(image, M, (round(width), round(height)))
    return image_rotated

if __name__ == '__main__':
    # os.system('gnome-screenshot -f test.png')
    image = cv.imread('test.png')
    image = image[:768, :1366]

    for i in range(0, 90, 10):
        plt.imshow(get_rotation(image, i))
        plt.show()

# plt.imshow(get)
# plt.imshow(image2)
# plt.show()
# cv.waitKey(0)