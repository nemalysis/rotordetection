import numpy as np

confusion_matrix_of_scenes = {}
confusion_matrix = np.zeros((4, 4)) #rows: true class, cols: detected class
confusion_matrix[0][0] = 200
confusion_matrix[0][1] = 0
confusion_matrix[0][2] = 0
confusion_matrix[0][3] = 0
confusion_matrix[1][0] = 0
confusion_matrix[1][1] = 206
confusion_matrix[1][2] = 0
confusion_matrix[1][3] = 0
confusion_matrix[2][0] = 0
confusion_matrix[2][1] = 0
confusion_matrix[2][2] = 212
confusion_matrix[2][3] = 0
confusion_matrix[3][0] = 11
confusion_matrix[3][1] = 13
confusion_matrix[3][2] = 9
confusion_matrix[3][3] = 169
confusion_matrix_of_scenes.update({'dusk': confusion_matrix})
confusion_matrix = np.zeros((4, 4)) #rows: true class, cols: detected class
confusion_matrix[0][0] = 101
confusion_matrix[0][1] = 0
confusion_matrix[0][2] = 0
confusion_matrix[0][3] = 113
confusion_matrix[1][0] = 0
confusion_matrix[1][1] = 2
confusion_matrix[1][2] = 0
confusion_matrix[1][3] = 209
confusion_matrix[2][0] = 0
confusion_matrix[2][1] = 0
confusion_matrix[2][2] = 63
confusion_matrix[2][3] = 136
confusion_matrix[3][0] = 0
confusion_matrix[3][1] = 0
confusion_matrix[3][2] = 3
confusion_matrix[3][3] = 209
confusion_matrix_of_scenes.update({'shadow': confusion_matrix})
confusion_matrix = np.zeros((4, 4)) #rows: true class, cols: detected class
confusion_matrix[0][0] = 203
confusion_matrix[0][1] = 0
confusion_matrix[0][2] = 0
confusion_matrix[0][3] = 8
confusion_matrix[1][0] = 0
confusion_matrix[1][1] = 206
confusion_matrix[1][2] = 0
confusion_matrix[1][3] = 0
confusion_matrix[2][0] = 0
confusion_matrix[2][1] = 0
confusion_matrix[2][2] = 213
confusion_matrix[2][3] = 6
confusion_matrix[3][0] = 0
confusion_matrix[3][1] = 9
confusion_matrix[3][2] = 8
confusion_matrix[3][3] = 197
confusion_matrix_of_scenes.update({'lowres': confusion_matrix})
confusion_matrix = np.zeros((4, 4)) #rows: true class, cols: detected class
confusion_matrix[0][0] = 5
confusion_matrix[0][1] = 0
confusion_matrix[0][2] = 0
confusion_matrix[0][3] = 198
confusion_matrix[1][0] = 0
confusion_matrix[1][1] = 0
confusion_matrix[1][2] = 0
confusion_matrix[1][3] = 207
confusion_matrix[2][0] = 0
confusion_matrix[2][1] = 0
confusion_matrix[2][2] = 3
confusion_matrix[2][3] = 204
confusion_matrix[3][0] = 5
confusion_matrix[3][1] = 0
confusion_matrix[3][2] = 0
confusion_matrix[3][3] = 238
confusion_matrix_of_scenes.update({'angle': confusion_matrix})
confusion_matrix = np.zeros((4, 4)) #rows: true class, cols: detected class
confusion_matrix[0][0] = 202
confusion_matrix[0][1] = 0
confusion_matrix[0][2] = 0
confusion_matrix[0][3] = 0
confusion_matrix[1][0] = 0
confusion_matrix[1][1] = 202
confusion_matrix[1][2] = 0
confusion_matrix[1][3] = 4
confusion_matrix[2][0] = 0
confusion_matrix[2][1] = 0
confusion_matrix[2][2] = 204
confusion_matrix[2][3] = 0
confusion_matrix[3][0] = 25
confusion_matrix[3][1] = 16
confusion_matrix[3][2] = 27
confusion_matrix[3][3] = 133
confusion_matrix_of_scenes.update({'IR': confusion_matrix})
confusion_matrix = np.zeros((4, 4)) #rows: true class, cols: detected class
confusion_matrix[0][0] = 196
confusion_matrix[0][1] = 0
confusion_matrix[0][2] = 0
confusion_matrix[0][3] = 3
confusion_matrix[1][0] = 0
confusion_matrix[1][1] = 199
confusion_matrix[1][2] = 0
confusion_matrix[1][3] = 0
confusion_matrix[2][0] = 0
confusion_matrix[2][1] = 0
confusion_matrix[2][2] = 199
confusion_matrix[2][3] = 0
confusion_matrix[3][0] = 11
confusion_matrix[3][1] = 15
confusion_matrix[3][2] = 14
confusion_matrix[3][3] = 181
confusion_matrix_of_scenes.update({'indoor': confusion_matrix})
confusion_matrix = np.zeros((4, 4)) #rows: true class, cols: detected class
confusion_matrix[0][0] = 199
confusion_matrix[0][1] = 0
confusion_matrix[0][2] = 0
confusion_matrix[0][3] = 3
confusion_matrix[1][0] = 0
confusion_matrix[1][1] = 195
confusion_matrix[1][2] = 0
confusion_matrix[1][3] = 6
confusion_matrix[2][0] = 0
confusion_matrix[2][1] = 0
confusion_matrix[2][2] = 196
confusion_matrix[2][3] = 4
confusion_matrix[3][0] = 22
confusion_matrix[3][1] = 11
confusion_matrix[3][2] = 3
confusion_matrix[3][3] = 182
confusion_matrix_of_scenes.update({'dark': confusion_matrix})
confusion_matrix = np.zeros((4, 4)) #rows: true class, cols: detected class
confusion_matrix[0][0] = 49
confusion_matrix[0][1] = 0
confusion_matrix[0][2] = 0
confusion_matrix[0][3] = 153
confusion_matrix[1][0] = 0
confusion_matrix[1][1] = 8
confusion_matrix[1][2] = 0
confusion_matrix[1][3] = 204
confusion_matrix[2][0] = 0
confusion_matrix[2][1] = 0
confusion_matrix[2][2] = 39
confusion_matrix[2][3] = 151
confusion_matrix[3][0] = 6
confusion_matrix[3][1] = 18
confusion_matrix[3][2] = 4
confusion_matrix[3][3] = 220
# confusion_matrix_of_scenes.update({'Rain_1280x720': confusion_matrix})
confusion_matrix_of_scenes.update({'rain': confusion_matrix})
confusion_matrix = np.zeros((4, 4)) #rows: true class, cols: detected class
confusion_matrix[0][0] = 197
confusion_matrix[0][1] = 0
confusion_matrix[0][2] = 0
confusion_matrix[0][3] = 11
confusion_matrix[1][0] = 0
confusion_matrix[1][1] = 96
confusion_matrix[1][2] = 5
confusion_matrix[1][3] = 100
confusion_matrix[2][0] = 0
confusion_matrix[2][1] = 0
confusion_matrix[2][2] = 153
confusion_matrix[2][3] = 54
confusion_matrix[3][0] = 6
confusion_matrix[3][1] = 15
confusion_matrix[3][2] = 18
confusion_matrix[3][3] = 157
confusion_matrix_of_scenes.update({'Rain_854x480': confusion_matrix})

if __name__ == '__main__':
    for scene in confusion_matrix_of_scenes.keys():
        print(scene)
        print(confusion_matrix_of_scenes[scene])