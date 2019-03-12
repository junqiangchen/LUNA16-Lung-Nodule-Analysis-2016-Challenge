import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from ResNet3d.model_resNet3d import ResNet3dModule
import numpy as np
import pandas as pd


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    # csv file should have the type:
    # label,data_npy
    # label,data_npy
    # ....
    #
    csvimagedata = pd.read_csv('dataprocess\\data\\nodel_all_train.csv')
    data = csvimagedata.iloc[:, :].values
    np.random.shuffle(data)
    # For Image
    images = data[:, 1:]
    # For Labels
    labels = data[:, 0]
    ResNet3d = ResNet3dModule(48, 48, 48, channels=1, n_class=2, costname="cross_entropy")
    ResNet3d.train(images, labels, "resnet.pd", "log\\NoudleClassfy\\resnet\\", 0.001, 0.7, 5, 32)


train()
