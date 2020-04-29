import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

import cv2
from Vnet.model_vnet3d import Vnet3dModule
import numpy as np
from Vnet.layer import save_images


def predict():
    src_path = "G:\Data\LIDC\LUNA16\segmentation\Image\\3_98\\"
    mask_path = "G:\Data\LIDC\LUNA16\segmentation\Mask\\3_98\\"
    imges = []
    masks = []
    for z in range(16):
        img = cv2.imread(src_path + str(z) + ".bmp", cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path + str(z) + ".bmp", cv2.IMREAD_GRAYSCALE)
        imges.append(img)
        masks.append(mask)

    test_imges = np.array(imges)
    test_imges = np.reshape(test_imges, (16, 96, 96))

    test_masks = np.array(masks)
    test_masks = np.reshape(test_masks, (16, 96, 96))
    Vnet3d = Vnet3dModule(96, 96, 16, channels=1, costname=("dice coefficient",), inference=True,
                          model_path="log\segmeation\model\Vnet3d.pd-50000")
    predict = Vnet3d.prediction(test_imges)
    test_images = np.multiply(test_imges, 1.0 / 255.0)
    test_masks = np.multiply(test_masks, 1.0 / 255.0)
    save_images(test_images, [4, 4], "test_src.bmp")
    save_images(test_masks, [4, 4], "test_mask.bmp")
    save_images(predict, [4, 4], "test_predict.bmp")


predict()
