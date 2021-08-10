


import os
import time
import csv
import pickle

import numpy as np
import cv2 as cv
import sklearn
from sklearn.metrics import accuracy_score

from pyzbar import pyzbar as zbar



from barcode.detection import detect_from_image
from barcode.rotation import get_rotation


def analyse_scene(scene_root):

    print('analysing scene ', scene_root)

    right_answers = 0
    wrong_answers = 0

    #results: (true_class, detected_class) -> count
    results = {}
    for true_class in range(1, 5):
        for detected_class in range(1, 5):
            results.update({(true_class, detected_class): 0})


    for true_class in range(1, 5): #class
        for _, _, files in os.walk('{}/{}'.format(scene_root, true_class)):
            for f in files:
                image = cv.imread('{}/{}/{}'.format(scene_root, true_class, f))
                detected_class = detect_from_image(image)

                #frames labeled store noise in 4
                if detected_class == 0:
                    detected_class = 4

                results.update({(true_class, detected_class): results[(true_class, detected_class)] + 1})
                # print('\r', scene_root, f, true_class, detected_class)

    pickle.dump(results, open('{}/results.dat'.format(scene_root), 'wb'))


def get_scenes(scenes_root):

    for root, dirs, files in os.walk(scenes_root):
        if '1' in dirs:
            yield root



def get_accuracy(results):
    y_true = []
    y_pred = []

    for true_class in range(1, 5):
        for detected_class in range(1, 5):
            for _ in range(results[(true_class, detected_class)]):
                y_true.append(true_class)
                y_pred.append(detected_class)

    return sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)


if __name__ == '__main__':

    scenes_root = './barcode/frames_labeled'

    # Get data for scenes
    for scene in get_scenes(scenes_root):
        analyse_scene(scene)

    # Print accuracy for scenes:
    for scene in get_scenes(scenes_root):
        # print(scene)
        result = pickle.load(open('{}/results.dat'.format(scene), 'rb'))
        print(scene, ':', get_accuracy(result))
