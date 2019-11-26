from __future__ import print_function, division
import numpy as np
import cv2
import os


def subimage_generator(image, mask, patch_block_size, numberxy, numberz):
    """
    generate the sub images and masks with patch_block_size
    :param image:
    :param patch_block_size:
    :param stride:
    :return:
    """
    width = np.shape(image)[1]
    height = np.shape(image)[2]
    imagez = np.shape(image)[0]
    block_width = np.array(patch_block_size)[1]
    block_height = np.array(patch_block_size)[2]
    blockz = np.array(patch_block_size)[0]
    stridewidth = (width - block_width) // numberxy
    strideheight = (height - block_height) // numberxy
    stridez = (imagez - blockz) // numberz
    # step 1:if stridez is bigger 1,return  numberxy * numberxy * numberz samples
    if stridez >= 1 and stridewidth >= 1 and strideheight >= 1:
        step_width = width - (stridewidth * numberxy + block_width)
        step_width = step_width // 2
        step_height = height - (strideheight * numberxy + block_height)
        step_height = step_height // 2
        step_z = imagez - (stridez * numberz + blockz)
        step_z = step_z // 2
        hr_samples_list = []
        hr_mask_samples_list = []
        for z in range(step_z, numberz * (stridez + 1) + step_z, numberz):
            for x in range(step_width, numberxy * (stridewidth + 1) + step_width, numberxy):
                for y in range(step_height, numberxy * (strideheight + 1) + step_height, numberxy):
                    if np.max(mask[z:z + blockz, x:x + block_width, y:y + block_height]) != 0:
                        hr_samples_list.append(image[z:z + blockz, x:x + block_width, y:y + block_height])
                        hr_mask_samples_list.append(mask[z:z + blockz, x:x + block_width, y:y + block_height])
        hr_samples = np.array(hr_samples_list).reshape((len(hr_samples_list), blockz, block_width, block_height))
        hr_mask_samples = np.array(hr_mask_samples_list).reshape(
            (len(hr_mask_samples_list), blockz, block_width, block_height))
        return hr_samples, hr_mask_samples
    # step 2:other sutitation,return one samples
    else:
        nb_sub_images = 1 * 1 * 1
        hr_samples = np.zeros(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float)
        hr_mask_samples = np.zeros(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float)
        rangz = min(imagez, blockz)
        rangwidth = min(width, block_width)
        rangheight = min(height, block_height)
        hr_samples[0, 0:rangz, 0:rangwidth, 0:rangheight] = image[0:rangz, 0:rangwidth, 0:rangheight]
        hr_mask_samples[0, 0:rangz, 0:rangwidth, 0:rangheight] = mask[0:rangz, 0:rangwidth, 0:rangheight]
        return hr_samples, hr_mask_samples


def make_patch(image, mask, patch_block_size, numberxy, numberz):
    """
    make number patch
    :param image:[depth,512,512]
    :param patch_block: such as[64,128,128]
    :return:[samples,64,128,128]
    expand the dimension z range the subimage:[startpostion-blockz//2:endpostion+blockz//2,:,:]
    """
    image_subsample, mask_subsample = subimage_generator(image=image, mask=mask, patch_block_size=patch_block_size,
                                                         numberxy=numberxy, numberz=numberz)
    return image_subsample, mask_subsample


def gen_image_mask(srcimg, seg_image, index, shape, numberxy, numberz, trainImage, trainMask):
    # step 2 get subimages (numberxy*numberxy*numberz,64, 128, 128)
    sub_srcimages,sub_liverimages = make_patch(srcimg,seg_image, patch_block_size=shape, numberxy=numberxy, numberz=numberz)
    # step 3 only save subimages (numberxy*numberxy*numberz,64, 128, 128)
    samples, imagez = np.shape(sub_srcimages)[0], np.shape(sub_srcimages)[1]
    for j in range(samples):
        sub_masks = sub_liverimages.astype(np.float32)
        sub_masks = np.clip(sub_masks, 0, 255).astype('uint8')
        if np.max(sub_masks[j, :, :, :]) == 255:
            filepath = trainImage + "\\" + str(index) + "_" + str(j) + "\\"
            filepath2 = trainMask + "\\" + str(index) + "_" + str(j) + "\\"
            if not os.path.exists(filepath) and not os.path.exists(filepath2):
                os.makedirs(filepath)
                os.makedirs(filepath2)
            for z in range(imagez):
                image = sub_srcimages[j, z, :, :]
                image = image.astype(np.float32)
                image = np.clip(image, 0, 255).astype('uint8')
                cv2.imwrite(filepath + str(z) + ".bmp", image)
                cv2.imwrite(filepath2 + str(z) + ".bmp", sub_masks[j, z, :, :])


def prepare3dtraindata(srcpath, maskpath, trainImage, trainMask, number, height, width, shape=(16, 256, 256),
                       numberxy=3, numberz=20):
    for i in range(number):
        index = 0
        listsrc = []
        listmask = []
        for _ in os.listdir(srcpath + str(i)):
            image = cv2.imread(srcpath + str(i) + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(maskpath + str(i) + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            listsrc.append(image)
            listmask.append(label)
            index += 1

        imagearray = np.array(listsrc)
        imagearray = np.reshape(imagearray, (index, height, width))
        maskarray = np.array(listmask)
        maskarray = np.reshape(maskarray, (index, height, width))
        gen_image_mask(imagearray, maskarray, i, shape=shape, numberxy=numberxy, numberz=numberz, trainImage=trainImage,
                       trainMask=trainMask)


def preparenoduledetectiontraindata():
    height = 512
    width = 512
    number = 601
    srcpath = "G:\Data\LIDC\LUNA16\process\image\\"
    maskpath = "G:\Data\LIDC\LUNA16\process\mask\\"
    trainImage = "G:\Data\LIDC\LUNA16\segmentation\Image"
    trainMask = "G:\Data\LIDC\LUNA16\segmentation\Mask"
    prepare3dtraindata(srcpath, maskpath, trainImage, trainMask, number, height, width, (16, 96, 96), 10, 16)


preparenoduledetectiontraindata()
