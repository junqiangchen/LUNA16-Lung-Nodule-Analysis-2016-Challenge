import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from ResNet3d.model_resNet3d import ResNet3dModule
import numpy as np
import pandas as pd


def predict():
    csvimagedata = pd.read_csv('dataprocess\\data\\nodel_all_test.csv')
    data = csvimagedata.iloc[:, :].values
    # For Image
    images = data[:, 1:]
    # For Labels
    labels = data[:, 0]
    ResNet3d = ResNet3dModule(48, 48, 48, channels=1, n_class=2, costname="cross_entropy", inference=True,
                              model_path="log\\NoudleClassfy\\resnet\model\\resnet.pd-50000")

    predictvalues = []
    predict_probs = []
    for num in range(np.shape(images)[0]):
        batchimage = np.reshape(np.load(images[num][0]), (1, 48, 48, 48, 1))
        predictvalue, predict_prob = ResNet3d.prediction(batchimage)
        predictvalues.append(predictvalue)
        predict_probs.append(predict_prob)

    name = 'classify_metrics.csv'
    out = open(name, 'w')
    out.writelines("y_predict" + "," + "y_score" + "," + "y_true" + "\n")
    labels = labels.tolist()
    for index in range(np.shape(images)[0]):
        out.writelines(
            str(predictvalues[index][0]) + "," + str(predict_probs[index][0]) + "," + str(labels[index]) + "\n")


predict()
