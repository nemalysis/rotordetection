# rotordetection
simplistic rotordetection with zbar

## Running the Code

To install required libraries run 'python -m pip install requirements.txt'
or if you have pip installed 'pip install requirements.txt'

## Barcode Algorithm

The Bot for analysing the RTSP livestream is in the ./barcode_bot.py script.
To calculate accuracies, run the ./barcode_analysis.py script.

## Deep Learning Algorithm

### Bot usage
To use the Deep Learning algorithm on an RTSP livestream, use `deep_learning_bot.py`. Before usage, first change the `RTSP_url` variable to the URL of your RTSP livestream.
After that, you can run the script. The script will open a window showing the livestream, together with the prediction of the algorithm overlayed on top of the image.

### Training
The code used for training the network can be found in `deep_learning/train_network.py`. A small sample of the training data is included in `deep_learning/data/labeled/` as a demo. 
The full dataset can be made available upon request. Before training the network, make sure to first run `deep_learning/data_preprocessing.py`.

### Preprocessing
The code used for preprocessing the data can be found in `deep_learning/data_preprocessing.py`. It uses the data contained in the `deep_learning/data/labeled/` folder. 
Make sure to navigate to the `deep_learning/` folder first and run it from there, such that the script can access the labeled data via `./data/labeled/`.

### Analysis
The code used for analysing the performance can be found in `deep_learning_analysis.py`.
