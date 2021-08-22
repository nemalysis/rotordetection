

import os
import cv2 as cv
import numpy as np
import pyautogui

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
RTSP_url = 'rtsp://operator:operator@192.168.0.90:554/axis-media/media.amp'
cap = cv.VideoCapture(RTSP_url)

i = 0
try:
    while(True):
        i += 1
        ret, image = cap.read()
        # image = image[:screenshot_clip_height, :screenshot_clip_width]
        # image_norm = cv.normalize(image, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        image_norm = image
        #if cv.waitKey(1) & 0xFF == ord('q'):
        #    break
        found = False
        start = time.time()
        # for deg in range(0, 90, 5):
        for deg in range(0, 90, 10):
            image_rotated = get_rotation(image_norm, deg)
            res = zbar.decode(image_rotated, [2, 5]) #EAN5
            if len(res) > 0:
                found = True
                # print("FOUND AT DEG {}; time: {}".format(deg, time.time() - start))
                # pp(res)
                break
        
        if not found:
            print('Its clouded :(')
        if res[0].data == b'33333':
            print('ROT 1')
        if res[0].data == b'66666':
            print('ROT 2')
        if res[0].data == b'99999':
            print('ROT 3')

except Exception:
    # image = pyautogui.screenshot()
    plt.imshow(image)
    plt.show()
    cv.waitKey(0)
    
cap.release()
cv.destroyAllWindows()