from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import os
from glob import glob
import pandas as pd

tqdm = lambda x: x


# get mask_region from csv data
def make_mask(mask, v_center, v_diam,spacing):
    v_diam_z=int(diam/spacing[2]+1)
    v_diam_y=int(diam/spacing[1]+1)
    v_diam_x=int(diam/spacing[0]+1)
    v_diam_z = np.rint(v_diam_z / 2)
    v_diam_y = np.rint(v_diam_y / 2)
    v_diam_x = np.rint(v_diam_x / 2)
    z_min = int(v_center[0] - v_diam_z)
    z_max = int(v_center[0] + v_diam_z + 1)
    x_min = int(v_center[1] - v_diam_x)
    x_max = int(v_center[1] + v_diam_x + 1)
    y_min = int(v_center[2] - v_diam_y)
    y_max = int(v_center[2] + v_diam_y + 1)
    mask[z_min:z_max, x_min:x_max, y_min:y_max] = 1.0
    # output nodule pixel size for preparing the classify
    print((z_max - z_min, x_max - x_min, y_max - y_min))


# Helper function to get rows in data frame associated with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return (f)


# Getting list of image files and save mask image files
for subsetindex in range(10):
    luna_path = "G:\Data\LIDC\LUNA16\LUNA16\src/"
    luna_subset_path = luna_path + "subset" + str(subsetindex) + "/"
    output_path = "G:\Data\LIDC\LUNA16\LUNA16\mask/"
    luna_subset_mask_path = output_path + "subset" + str(subsetindex) + "/"
    if not os.path.exists(luna_subset_mask_path):
        os.makedirs(luna_subset_mask_path)
    file_list = glob(luna_subset_path + "*.mhd")
    
    file_list_path=[]
    for i in range(len(file_list)):
        file_list_path.append(file_list[i][0:-4])

    # The locations of the nodes
    luna_csv_path = "G:\Data\LIDC\LUNA16"
    df_node = pd.read_csv(luna_csv_path + "/CSVFILES/" + "annotations.csv")
    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list_path, file_name))
    df_node = df_node.dropna()
    # Looping over the image files
    for fcount, img_file in enumerate(tqdm(file_list_path)):
        # get all nodules associate with file
        mini_df = df_node[df_node["file"] == img_file]
        # load the src data once
        img_file=img_file+".mhd"
        itk_img = sitk.ReadImage(img_file)
        # indexes are z,y,x (notice the ordering)
        img_array = sitk.GetArrayFromImage(itk_img)
        # num_z height width constitute the transverse plane
        num_z, height, width = img_array.shape
        # x,y,z  Origin in world coordinates (mm)
        origin = np.array(itk_img.GetOrigin())
        # spacing of voxels in world coor. (mm)
        spacing = np.array(itk_img.GetSpacing())
        # some files may not have a nodule--skipping those
        if mini_df.shape[0] == 0:
            # set out mask data once
            mask_itk = np.zeros(shape=(num_z, height, width), dtype=np.float)
        if mini_df.shape[0] > 0:
            # set out mask data once
            mask_itk = np.zeros(shape=(num_z, height, width), dtype=np.float)
            # go through all nodes in one series image
            for node_idx, cur_row in mini_df.iterrows():
                node_x = cur_row["coordX"]
                node_y = cur_row["coordY"]
                node_z = cur_row["coordZ"]
                diam = cur_row["diameter_mm"]
                center = np.array([node_x, node_y, node_z])
                # nodule center
                v_center = np.rint((center - origin) / spacing)
                # nodule diam
                v_diam = diam
                # convert x,y,z order v_center to z,y,z order v_center
                v_center[0], v_center[1], v_center[2] = v_center[2], v_center[1], v_center[0]
                make_mask(mask_itk, v_center, v_diam,spacing)

          mask_itk = np.uint8(mask_itk * 255.)
          mask_itk = np.clip(mask_itk, 0, 255).astype('uint8')
          sitk_maskimg = sitk.GetImageFromArray(mask_itk)
          sitk_maskimg.SetSpacing(spacing)
          sitk_maskimg.SetOrigin(origin)
          sub_img_file = img_file[len(luna_subset_path):-4]
          sitk.WriteImage(sitk_maskimg, luna_subset_mask_path + sub_img_file + "_segmentation.mhd")
