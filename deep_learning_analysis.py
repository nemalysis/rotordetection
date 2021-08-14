import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_conditions_names2keys():
    # names of the videos and their respective keys in the analysis
    conditions_names2keys = {
        'baseline': 'indoor',
        'daemmerung_leicht': 'dusk',
        'daemmerung_stark': 'dark',
        'IR': 'IR',
        'low_res': 'lowres',
        'regen': 'rain',
        'schatten': 'shadow',
        'winkel': 'angle',
    }

    return conditions_names2keys


def compute_accuracy(model,test_dir,img_size,conditions_names2keys):
    test_results = {}
    for test_condition in os.listdir(test_dir):
        test_set_dir = os.path.join(test_dir,test_condition)
        print(test_condition)

        test_generator = test_datagen.flow_from_directory(
            test_set_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical"
        )
        
        curr_test_results = model.evaluate(test_generator)
        test_results[test_condition] = curr_test_results

    n_cond = len(conditions_names2keys)
    acc_DL = [0 for i in range(n_cond)]
    label = ['' for i in range(n_cond)]

    for count, condition in enumerate(conditions_names2keys.keys()):
        acc_DL[count] = test_results[condition][1]
        label[count] = conditions_names2keys[condition]

    return acc_DL,label


def plot_accuracy(acc_DL,label,conditions_names2keys):
    n_cond = len(conditions_names2keys)
    # mpl.rcdefaults()
    plt.rcParams.update({'font.size': 14})
    fig1, ax1 = plt.subplots()
    x_plt = np.arange(n_cond,0,-1)
    ax1.grid(True)
    ax1.set_axisbelow(True)
    ax1.barh(x_plt,acc_DL,facecolor='rebeccapurple')
    plt.yticks(x_plt,label)
    plt.ylabel('condition')
    plt.xlabel('accuracy')
    plt.title('Performance CNN')

    for i, y in enumerate(acc_DL):
        ax1.text(y-0.01, n_cond-i, '{}%'.format(np.round(y*100,2)), color='white', fontweight='bold',verticalalignment='center',horizontalalignment='right')

    plt.savefig(r'.\deep_learning\analysis_results\accuracy\accuracy_'+model_name+'.svg',bbox_inches='tight')
    plt.savefig(r'.\deep_learning\analysis_results\accuracy\accuracy_'+model_name+'.pdf',bbox_inches='tight')


def compute_conf_mats(model,test_dir,img_size,batch_size):
    # Confusion Matrix
    confusion_mats = {}

    for test_condition in os.listdir(test_dir):
        test_set_dir = os.path.join(test_dir,test_condition)
        print(test_condition + ':')
        if 'merged' in test_condition:
            continue
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            test_set_dir,
            label_mode = 'categorical',
            smart_resize = True,
            image_size=img_size,
            batch_size=batch_size)

        y_pred = np.array([])
        y_gt =  np.array([])
        for x, y in test_ds:
            y_pred = np.concatenate([y_pred, np.argmax(model.predict(x),axis=-1)])
            y_gt = np.concatenate([y_gt, np.argmax(y.numpy(), axis=-1)])

        confusion_mats[test_condition] = tf.math.confusion_matrix(labels=y_gt, predictions=y_pred).numpy()
    
    return confusion_mats


def plot_confusion_mats(confusion_mats,conditions_names2keys):
    plt.figure(figsize=(14,14))
    for i,cond in enumerate(conditions_names2keys.keys()):
        plt.subplot(3,3,i+1)
        
        sn.heatmap(confusion_mats[cond], annot=True,square=True,xticklabels=['DL1','DL2','DL3','DL_BG'],yticklabels=['DL1','DL2','DL3','DL_BG'],cbar_kws={"shrink":0.8})
        plt.ylabel('ground truth')
        plt.xlabel('prediction')
        plt.title(conditions_names2keys[cond])

    plt.savefig(r'.\deep_learning\analysis_results\confusion_mat\confusion_mat.pdf',bbox_inches='tight')


if __name__ == "main":

    batch_size = 5
    img_height = 360
    img_width = 640

    model_dir = r'.\deep_learning\model\rotordet_net_v4'
    model_name = os.path.basename(model_dir)
    model = tf.keras.models.load_model(model_dir)

    test_datagen = ImageDataGenerator()
    test_dir = r'.\deep_learning\data\test'
    img_size = (img_height, img_width)

    conditions_names2keys = get_conditions_names2keys()

    acc_DL,label = compute_accuracy(model,test_dir,img_size,conditions_names2keys)
    plot_accuracy(acc_DL,label,conditions_names2keys)

    confusion_mats = compute_conf_mats(model,test_dir,img_size,batch_size)
    plot_confusion_mats(confusion_mats,conditions_names2keys)

