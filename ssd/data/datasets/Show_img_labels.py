# coding: utf-8
import os
import torch.utils.data
import numpy as np
import visdrone
from torch.utils.data.dataloader import default_collate
# print(default_collate)
import cv2
dataset_dir = '/media/e813/E/dataset/eccv/eccv/VisDrone2018-DET-train'
visdrone_dataset = visdrone.VisDroneDataset(dataset_dir)
all_flag = False
for image,target,i in visdrone_dataset:
    print(i)
    exist_others_flag =False
    for n in range(target['boxes'].shape[0]):
        if target['labels'][n] == 0:
            # image = cv2.rectangle(image, (target['boxes'][n, 0], target['boxes'][n, 1]),
            #                       (target['boxes'][n, 2], target['boxes'][n, 3]), (255, 0, 0))
            exist_others_flag =True
            all_flag =True
            # print('there finded the target!')
            # exit()
    if exist_others_flag:
        pass
        # cv2.imshow('0', image)
        # cv2.waitKey(3000)
    continue
    for n in range(target['boxes'].shape[0]):
        image = cv2.rectangle(image,(target['boxes'][n,0],target['boxes'][n,1]),(target['boxes'][n,2],target['boxes'][n,3]),(255,0,0))
    cv2.imshow('0',image)
    cv2.waitKey(1)
    # break
# VOCdataset = VOCDataset('/media/e813/E/dataset/VOCdevkit/VOC2007',)
print('all flag is ',all_flag)