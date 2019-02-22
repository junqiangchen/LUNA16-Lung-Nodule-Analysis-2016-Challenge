# LUNA16-LUng-Nodule-Analysis-2016-Challenge
> This is an example of the CT images lung nodule detection and false positive reduction from LUNA16-LUng-Nodule-Analysis-2016-Challenge
![](luna16_header.png)

## How to Use

**1、Preprocess**
* convert annotation.csv file to image mask file:run the LUNA_mask_extraction.py
* analyze the ct image,and get the slice thickness and window width and position:run the dataAnaly.py
* get lung nodule ct image and mask:run the data2dprepare.py
* get patch(96,96,16) lung nodule image and mask:run the data3dprepare.py
* convert candidates.csv file to nodule and not-nodule image(48,48,48):run the LUNA_node_extraction.py
* Augment the nodule image data:run the Augmain.py
* split data into train data(80%) and test data(20%):run the subset.py

**2、Nodule Detection**
* the VNet model

![](3dVNet.png) 

* train and predict in the script of vnet3d_train.py and vnet3d_predict.py

**3、False Positive Reducution**
* the ResVGGNet model

![](ResVGGNet.png)

* train and predict in the script of ResNet3d_train.py and ResNet3d_predict.py

**4、trained model can download on here:**

## Result


## Contact
* https://github.com/junqiangchen
* email: 1207173174@qq.com
* Contact: junqiangChen
* WeChat Number: 1207173174
* WeChat Public number: 最新医学影像技术
