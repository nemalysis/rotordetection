

import os
from pprint import pprint as pp

import os
import cv2 as cv
import numpy as np


import matplotlib.pyplot as plt
from pyzbar import pyzbar as zbar
from pprint import pprint as pp

from barcode.rotation import get_rotation
import time

import csv

def detect_from_image(image):

    image_norm = image
    found = False
    frame_res = None
    start = time.time()
    for deg in range(0, 100, 10):
        image_rotated = get_rotation(image_norm, deg)
        # image_rotated = image_norm
        res = zbar.decode(image_rotated, [5])  # EAN5
        # res = []
        if len(res) > 0:
            found = True
            # print("FOUND AT DEG {}; time: {}".format(deg, time.time() - start))
            # pp(res)
            break
            pass

    if not found:
        for deg in range(-90, 0, 10):
            image_rotated = get_rotation(image_norm, deg)
            # image_rotated = image_norm
            res = zbar.decode(image_rotated, [5])  # EAN5
            # res = []
            if len(res) > 0:
                found = True
                # print("FOUND AT DEG {}; time: {}".format(deg, time.time() - start))
                # pp(res)
                break
                pass


    # print(time.time() - start)

    if not found:
        frame_res = 0
    else:
        if not res[0].data in [b'33333', b'66666', b'99999']:
            frame_res = 0
        if res[0].data == b'33333':
            frame_res = 1
        if res[0].data == b'66666':
            frame_res = 2
        if res[0].data == b'99999':
            frame_res = 3

    return frame_res


def detect_from_file(filename):

    csvname = filename.replace('mp4', 'csv')
    csvfile = open(csvname, 'w')

    vc = cv.VideoCapture(filename)
    ret, image = vc.read()
    frame_count = 0

    length = int(vc.get(cv.CAP_PROP_FRAME_COUNT))
    while ret == True:
        frame_count += 1

        frame_res = detect_from_image(image)

        print('\r', frame_count/length, frame_res, end='')
        csvfile.write('{}, {}\n'.format(frame_count, frame_res))
        csvfile.flush()
        ret, image = vc.read()
    print()
    csvfile.close()


def plot_frequency(filename):

    filename = filename.replace('mp4', 'csv')
    plotname = filename.replace('csv', 'png')

    data = np.loadtxt(filename, delimiter=',')
    # pp(data)

    frames = data[:, 0]
    detected = data[:, 1]

    rot_1 = np.zeros(len(detected))
    rot_2 = np.zeros(len(detected))
    rot_3 = np.zeros(len(detected))

    rot_1[detected == 1] = 1
    rot_2[detected == 2] = 1
    rot_3[detected == 3] = 1

    plt.figure()
    plt.title(plotname)
    plt.plot(rot_1, label='ROT1')
    plt.plot(rot_2, label='ROT2')
    plt.plot(rot_3, label='ROT3')

    plt.legend()
    plt.savefig(plotname)
    # plt.show()
    pass

if __name__ == '__main__':

    for root, dirs, files in os.walk('./videos'):
        if len(files) == 0:
            print('No Video in ', root)
            continue

        for f in files:
            filename = '{}/{}'.format(root, f)
            if not filename.endswith('mp4'):
                continue
            print('Analysing Video: ', filename)
            if not os.path.isfile(filename.replace('mp4', 'csv')):
                detect_from_file(filename)
            plot_frequency(filename)



    # detect_from_file("./videos/schatten/1280x720/50fps/draussen_1280x720_50fps.mp4")
    # plot_frequency("./videos/schatten/1280x720/50fps/draussen_1280x720_50fps.mp4")
