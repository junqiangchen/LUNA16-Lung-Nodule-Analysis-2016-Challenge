from __future__ import print_function, division
import numpy as np
import cv2
import os


def subimage_generator(image, patch_block_size, numberxy, numberz):
    """
    generate the sub images with patch_block_size
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

    stridewidth = (width - block_width) // (numberxy - 1)
    strideheight = (height - block_height) // (numberxy - 1)
    fix_stridez = numberz // 2
    stridez = (imagez - blockz) // fix_stridez
    # step 1:if image size of z is smaller than blockz,return zeros samples
    if imagez < blockz:
        nb_sub_images = numberxy * numberxy * 1
        hr_samples = np.zeros(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float32)
        indx = 0
        for x in range(0, width - block_width + 1, stridewidth):
            for y in range(0, height - block_height + 1, strideheight):
                hr_samples[indx, 0:imagez, :, :] = image[:, x:x + block_width, y:y + block_height]
                indx += 1
        if (indx != nb_sub_images):
            print(indx)
            print(nb_sub_images)
            raise ValueError("error sub number image")
        return hr_samples
    # step 2:if stridez is bigger 1,return  numberxy * numberxy * numberz samples
    if stridez >= 1:
        nb_sub_images = numberxy * numberxy * (stridez + 1)
        hr_samples = np.empty(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float32)
        indx = 0
        for z in range(0, fix_stridez * (stridez + 1), fix_stridez):
            for x in range(0, width - block_width + 1, stridewidth):
                for y in range(0, height - block_height + 1, strideheight):
                    hr_samples[indx, :, :, :] = image[z:z + blockz, x:x + block_width, y:y + block_height]
                    indx += 1

        if (indx != nb_sub_images):
            print(indx)
            print(nb_sub_images)
            raise ValueError("error sub number image")
        return hr_samples

    # step3: if stridez==imagez,return numberxy * numberxy * 1 samples,one is [0:blockz,:,:]
    if imagez == blockz:
        nb_sub_images = numberxy * numberxy * 1
        hr_samples = np.empty(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float32)
        indx = 0
        for x in range(0, width - block_width + 1, stridewidth):
            for y in range(0, height - block_height + 1, strideheight):
                hr_samples[indx, :, :, :] = image[:, x:x + block_width, y:y + block_height]
                indx += 1
        if (indx != nb_sub_images):
            print(indx)
            print(nb_sub_images)
            raise ValueError("error sub number image")
        return hr_samples
    # step4: if stridez==0,return numberxy * numberxy * 2 samples,one is [0:blockz,:,:],two is [-blockz-1:-1,:,:]
    if stridez == 0:
        nb_sub_images = numberxy * numberxy * 2
        hr_samples = np.empty(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float32)
        indx = 0
        for x in range(0, width - block_width + 1, stridewidth):
            for y in range(0, height - block_height + 1, strideheight):
                hr_samples[indx, :, :, :] = image[0:blockz, x:x + block_width, y:y + block_height]
                indx += 1
                hr_samples[indx, :, :, :] = image[-blockz - 1:-1, x:x + block_width, y:y + block_height]
                indx += 1
        if (indx != nb_sub_images):
            print(indx)
            print(nb_sub_images)
            raise ValueError("error sub number image")
        return hr_samples


def make_patch(image, patch_block_size, numberxy, numberz):
    """
    make number patch
    :param image:[depth,512,512]
    :param patch_block: such as[64,128,128]
    :return:[samples,64,128,128]
    expand the dimension z range the subimage:[startpostion-blockz//2:endpostion+blockz//2,:,:]
    """
    image_subsample = subimage_generator(image=image, patch_block_size=patch_block_size, numberxy=numberxy,
                                         numberz=numberz)
    return image_subsample


def gen_image_mask(srcimg, seg_image, index, shape, numberxy, numberz, trainImage, trainMask):
    # step 2 get subimages (numberxy*numberxy*numberz,64, 128, 128)
    sub_srcimages = make_patch(srcimg, patch_block_size=shape, numberxy=numberxy, numberz=numberz)
    sub_liverimages = make_patch(seg_image, patch_block_size=shape, numberxy=numberxy, numberz=numberz)
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
