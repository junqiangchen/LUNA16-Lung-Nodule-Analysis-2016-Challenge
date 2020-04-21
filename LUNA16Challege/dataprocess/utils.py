import os


def file_name_path(file_dir):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs):
            print("sub_dirs:", dirs)
            return dirs

        
def files_name_path(file_dir):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    for root, dirs, files in os.walk(file_dir):
        if len(files):
            print("sub_files:", files)
            return files
        
def save_file2csvv2(file_dir, file_name,label):
    """
    save file path to csv,this is for classification
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :param label:classification label
    :return:
    """
    out = open(file_name, 'w')
    sub_files = files_name_path(file_dir)
    out.writelines("class,filename" + "\n")
    for index in range(len(sub_dirs)):
        out.writelines(file_dir + "/" + label+","+sub_files[index] + "\n")
       

def save_file2csv(file_dir, file_name):
    """
    save file path to csv,this is for segmentation
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    sub_dirs = file_name_path(file_dir)
    out.writelines("filename" + "\n")
    for index in range(len(sub_dirs)):
        out.writelines(file_dir + "/" + sub_dirs[index] + "\n")


#save_file2csv("G:\Data\LIST\\3dliver_25625616\Image", "train_X.csv")
