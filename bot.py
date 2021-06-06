

import os
import cv2 as cv
import numpy as np
import pyautogui
import PIL

import matplotlib.pyplot as plt
from pyzbar import pyzbar as zbar
from pprint import pprint as pp

from rotation import get_rotation
import time


#params
screenshot_cmd = 'gnome-screenshot -f test.png'
screenshot_clip_width = 1366
screenshot_clip_height = 768


image = None
image_rotated = None
try:
    while(True):
        os.system(screenshot_cmd)
        image = cv.imread('./test.png')
        image = image[:screenshot_clip_height, :screenshot_clip_width]
        # image_norm = cv.normalize(image, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        image_norm = image
        found = False
        start = time.time()
        # for deg in range(0, 90, 5):
        for deg in range(0, 90, 10):
            image_rotated = get_rotation(image_norm, deg)
            res = zbar.decode(image_rotated, [2, 5]) #EAN5
            if len(res) > 0:
                found = True
                print("FOUND AT DEG {}; time: {}".format(deg, time.time() - start))
                pp(res)
                break

        if not found:
            print('Its clouded :(')

except Exception:
    # image = pyautogui.screenshot()
    plt.imshow(image)
    plt.show()
    cv.waitKey(0)