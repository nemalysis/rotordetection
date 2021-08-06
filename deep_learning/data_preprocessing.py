import os
from pathlib import Path
import numpy as np
from PIL import Image
from tensorflow import keras
import tensorflow as tf
import random
from shutil import copyfile

def rot90(img,k,img_height,img_width):
    # img_out = [np.asarray(img),]
    img_out = np.asarray(img)
    img_out = tf.image.rot90(img_out, k=k)
    img_out = tf.image.resize_with_pad(img_out, img_height, img_width)
    img_out = np.asarray(img_out)
    # img_out = img_out[0,:,:,:]
    img_out = Image.fromarray(np.uint8(img_out),'RGB')
    return img_out

path_labeled_data = r'.\deep_learning\data\labeled'

outputfolder_train = r'.\deep_learning\data\train'
outputfolder_train_merged = r'.\deep_learning\data\merged'
Path(os.path.join(outputfolder_train_merged,'DL1')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(outputfolder_train_merged,'DL2')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(outputfolder_train_merged,'DL3')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(outputfolder_train_merged,'DL_BG')).mkdir(parents=True, exist_ok=True)

outputfolder_test = r'.\deep_learning\data\test'
outputfolder_test_merged = r'.\deep_learning\data\test\merged'
Path(outputfolder_test_merged).mkdir(parents=True, exist_ok=True)

condition_folders = [os.path.join(path_labeled_data,x) for x in os.listdir(path_labeled_data)] 
condition_folders = [os.path.join(path_labeled_data,x) for x in condition_folders if os.path.isdir(x)]

img_height = 360
img_width = 640

for i_condition, condition_folder in enumerate(condition_folders):
    class_folders = [os.path.join(condition_folder,x) for x in os.listdir(condition_folder)] 
    class_folders = [os.path.join(condition_folder,x) for x in class_folders if os.path.isdir(x)]
    print(os.path.basename(condition_folder))
    for class_folder in class_folders:
        print(os.path.basename(class_folder))
        imgs = [os.path.join(class_folder,x) for x in os.listdir(class_folder) if x.endswith('.png')]
        for img_sample in imgs:
            img = keras.preprocessing.image.load_img(img_sample, target_size=(img_height, img_width))
            curr_name = img_sample.split('\\')[-1]
            curr_class = img_sample.split('\\')[-2]
            curr_condition = img_sample.split('\\')[-3]
            curr_outputfolder = os.path.join(outputfolder_train,curr_condition,curr_class)
            Path(curr_outputfolder).mkdir(parents=True, exist_ok=True)
            out_name = os.path.join(curr_outputfolder, str(i_condition) + '_' + curr_name)
            img.save(out_name,format='PNG')


# List all folders in train folder
conditions_train = [os.path.join(outputfolder_train,x) for x in os.listdir(outputfolder_train)] 
conditions_train = [os.path.join(outputfolder_train,x) for x in conditions_train if os.path.isdir(x)]
exists_in_test = os.listdir(outputfolder_test)
for condition_folder_train in conditions_train:
    print(os.path.basename(condition_folder_train) + ':')
    if os.path.basename(condition_folder_train) in exists_in_test:
        print('skip')
        continue
    if 'merged' in condition_folder_train:
        print('skip')
        continue
    # List all class folders in current condition
    class_folders = [os.path.join(condition_folder_train,x) for x in os.listdir(condition_folder_train)] 
    class_folders = [os.path.join(condition_folder_train,x) for x in class_folders if os.path.isdir(x)]

    for class_folder in class_folders:
        curr_class = os.path.basename(class_folder)
        print(curr_class)
        # List all images in current class
        imgs = [os.path.join(class_folder,x) for x in os.listdir(class_folder) if x.endswith('.png')]
        n_imgs = len(imgs)
        
        # Randomly move images to test folder, such that 200 remain for training
        if n_imgs>=400:
            i_tomove = random.sample(range(n_imgs),n_imgs-200)
        else: # Or if there are less than 400 in total, take half of them
            i_tomove = random.sample(range(n_imgs),round(n_imgs/2))

        # Move the test images. Copy also into a merged folder to use as validation dataset
        imgs_tomove = [imgs[x] for x in i_tomove]
        print('Moving ' + str(len(imgs_tomove)) + ' images to test folder...')
        for img_tomove in imgs_tomove:
            curr_fname = img_tomove.split('\\')[-1]
            curr_fclass = img_tomove.split('\\')[-2]
            curr_fcondition = img_tomove.split('\\')[-3]
            
            # Get outputfolders right
            curr_outputfolder = os.path.join(outputfolder_test,curr_fcondition,curr_fclass)
            curr_outputfolder_merged = os.path.join(outputfolder_test_merged,curr_fclass)

            # Check for outputfolders
            Path(curr_outputfolder).mkdir(parents=True, exist_ok=True)
            Path(curr_outputfolder_merged).mkdir(parents=True, exist_ok=True)

            img_test_sorted = os.path.join(curr_outputfolder,curr_fname)
            img_test_merged = os.path.join(curr_outputfolder_merged,curr_fname)

            # Move to test folder
            os.rename(img_tomove,img_test_sorted)
            # Copy to merged test folder
            copyfile(img_test_sorted,img_test_merged)
        
        
        # Perform augmentation on remaining training samples
        print('Performing Data Augmentation...')
        imgs = [os.path.join(class_folder,x) for x in os.listdir(class_folder) if x.endswith('.png')]
        for img_sample in imgs:
            img = keras.preprocessing.image.load_img(img_sample, target_size=(img_height, img_width))
            img_savename_merged = os.path.join(outputfolder_train_merged, curr_class, os.path.basename(img_sample))
            img.save(img_savename_merged,format='PNG')

            img_rot90 = rot90(img,1,img_height,img_width)
            img_rot90_savename = img_sample.replace('.png','_rot90.png')
            img_rot90_savename_merged = os.path.join(outputfolder_train_merged, curr_class, os.path.basename(img_rot90_savename))
            img_rot90.save(img_rot90_savename_merged,format='PNG')

            img_rot180 = rot90(img,2,img_height,img_width)
            img_rot180_savename = img_sample.replace('.png','_rot180.png')
            img_rot180_savename_merged = os.path.join(outputfolder_train_merged, curr_class, os.path.basename(img_rot180_savename))
            img_rot180.save(img_rot180_savename_merged,format='PNG')

            img_rot270 = rot90(img,3,img_height,img_width)
            img_rot270_savename = img_sample.replace('.png','_rot270.png')
            img_rot270_savename_merged = os.path.join(outputfolder_train_merged, curr_class, os.path.basename(img_rot270_savename))
            img_rot270.save(img_rot270_savename_merged,format='PNG')
        print('Done.')

