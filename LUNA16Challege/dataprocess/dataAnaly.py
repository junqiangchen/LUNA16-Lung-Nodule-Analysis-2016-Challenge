from __future__ import print_function, division
import SimpleITK as sitk
from glob import glob

luna_path = "G:\Data\LIDC\LUNA16\LUNA16\src/"
output_path = "G:\Data\LIDC\LUNA16\LUNA16\mask/"


def getTrunctedThresholdValue():
    """
    remove outside of liver region value,and expand the tumor range when normalization 0 to 1.
    calculate the overlap between liver mask and src image with range of lower and upper value.
    :return:None
    """
    upper = 600
    lower = -1000
    num_points = 0.0
    num_inliers = 0.0
    for subsetindex in range(0, 10, 1):
        luna_subset_path = luna_path + "subset" + str(subsetindex) + "/"
        file_list = glob(luna_subset_path + "*.mhd")
        for fcount in range(len(file_list)):
            src = sitk.ReadImage(file_list[fcount], sitk.sitkInt16)
            srcimg = sitk.GetArrayFromImage(src)
            luna_subset_mask_path = output_path + "subset" + str(subsetindex) + "/"
            sub_img_file = file_list[fcount][len(luna_subset_path):-4]
            seg = sitk.ReadImage(luna_subset_mask_path + sub_img_file + "_segmentation.mhd", sitk.sitkUInt8)
            segimg = sitk.GetArrayFromImage(seg)
            seg_maskimage = segimg.copy()
            seg_maskimage[segimg > 1] = 255

            inliers = 0
            num_point = 0
            for z in range(seg_maskimage.shape[0]):
                for y in range(seg_maskimage.shape[1]):
                    for x in range(seg_maskimage.shape[2]):
                        if seg_maskimage[z][y][x] != 0:
                            num_point += 1
                            if (srcimg[z][y][x] < upper) and (srcimg[z][y][x] > lower):
                                inliers += 1
            # if not seg mask,not calculate
            if num_point != 0:
                print('{:.4}%'.format(inliers / num_point * 100))
                num_points += num_point
                num_inliers += inliers
    print(num_inliers / num_points)


def getitkImageSpacing():
    """
    get src itk image size and spacing,spacing value from 0.6 to 5,should resample image to have same z spacing.
    :return:None
    """
    for subsetindex in range(0, 10, 1):
        luna_subset_path = luna_path + "subset" + str(subsetindex) + "/"
        file_list = glob(luna_subset_path + "*.mhd")
        for fcount in range(len(file_list)):
            src = sitk.ReadImage(file_list[fcount], sitk.sitkInt16)
            srcSize = src.GetSize()
            srcSpace = src.GetSpacing()
            print(srcSize)
            print(srcSpace)


#getTrunctedThresholdValue()
# getitkImageSpacing()
