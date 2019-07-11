from __future__ import print_function, division
from glob import glob


# save .npy file path to csv
def save_npy2csv(path, name, labelnum=1):
    """
    this is for classify,save label+filepath into csv
    """
    out = open(name, 'w')
    file_list = glob(path + "*.npy")
    out.writelines("index" + "," + "filename" + "\n")
    for index in range(len(file_list)):
        out.writelines(str(labelnum) + "," + file_list[index] + "\n")


#save_npy2csv("G:\Data\LIDC\LUNA16\classsification\\1_aug\\", "nodel_positive.csv", 1)
