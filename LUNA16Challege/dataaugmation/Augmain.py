from dataaugmation.Augmentation.ImageAugmentation import DataAug3D

aug = DataAug3D(rotation=45, width_shift=0.05, height_shift=0.05, depth_shift=0, zoom_range=0)
aug.DataAugmentation('Train_X.csv', 40, aug_path='G:\Data\LIDC\LUNA16\classsification\\1_aug\\')
