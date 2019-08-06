# coding: utf-8
import os
import torch.utils.data
import numpy as np
from PIL import Image
import cv2

from ssd.structures.container import Container
'''
ignored regions (0), pedestrian (1), people (2), bicycle (3), car (4), van (5), truck (6), tricycle (7), awning-tricycle (8), bus (9), motor (10), others (11)
'''
class VisDroneDataset(torch.utils.data.Dataset):
    class_names = ('__background__', 'pedestrian', 'people', 'bicycle',
                    'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others')
    def __init__(self, dataset_dir, transform=None, target_transform=None):
        # as you would do normally
        self.dataset_dir = dataset_dir
        self.anno_list = os.listdir(os.path.join(dataset_dir,'annotations'))
        self.img_dir = os.path.join(dataset_dir,'images')
        self.transform = transform
        self.target_transform = target_transform
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.keep_difficult = True

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, index):
        img_name = self.anno_list[index].split('.')[0]+'.jpg'
        img_path = os.path.join(self.img_dir,img_name)
        image = self._read_image(img_path)
        boxes, labels, is_difficult = self._get_annotation(index)
        # load the bounding boxes in x1, y1, x2, y2 order.
        # boxes = np.array((N, 4), dtype=np.float32)
        # and labels
        # labels = np.array((N, ), dtype=np.int64)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        return image, targets, index

    def _read_image(self, img_path):
        # image = Image.open(img_path).convert("RGB")
        # image = np.array(image)
        image = cv2.imread(img_path)
        return image

    def _get_annotation(self,index):
        anno_path = os.path.join(self.dataset_dir,'annotations',self.anno_list[index])
        boxes = []
        labels = []
        is_difficult = []
        with open(anno_path,'r') as f:
            for line in f.readlines():
                obj = line.split(',')
                if obj[4]=='0':
                    continue
                x1, y1 = float(obj[0]),float(obj[1])
                x2, y2 = x1+float(obj[2]),y1+float(obj[3])
                boxes.append([x1, y1, x2, y2])
                labels.append(int(obj[5]))
                is_difficult.append(0)
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

