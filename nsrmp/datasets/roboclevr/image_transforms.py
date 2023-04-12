#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   image_transforms.py
#Time    :   2022/06/10 21:02:41
#Author  :   Namasivayam K
#Contact :   namasivayam.k@cse.iitd.ac.in
#Desc    :   This fle is a part of NSRM-PyTorch Release

#TODO: @Namas. The current implementation uses list and might be inefficient. Change this to work with numpy arrays

import cv2
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image

class Compose(transforms.Compose):
    def __call__(self,img,bbox):
        for t in self.transforms:
            img,bbox = t(img,bbox)
        return img,bbox

class ChangeBoxMode(object):
    def __init__(self,mode):
        self.mode = mode

    def __call__(self,img,bbox):
        '''
        input: bbox in any mode
        return: bbox in [x1,y1,x2,y2]
        '''
        if self.mode == 'yxhw':
            for i,b in enumerate(bbox):
                y1,x1,yh,xw,d = b
                #[y1,x1,h,w] to [x1,y1,x2,y2]
                bbox[i] = [x1,y1,x1+xw,y1+yh,d]

        elif self.mode == 'xywh':
            raise NotImplementedError
        else:
            raise NotImplementedError
        return img, bbox

class DenormalizeBbox(object):
    def __call__(self,img,bbox):
        '''
        img: PIL image
        bbox: bbox(normalized) in [x1,y1,x2,y2] format 
        '''
        assert isinstance(img, Image.Image) == True, "Image should be a PIL image" 
        w,h = img.size
        for i,b in enumerate(bbox):
            x1,y1,x2,y2,d = b
            bbox[i] = list(map(int,[x1*w, y1*h, x2*w, y2*h]))
            bbox[i].append(d)
        return img, bbox

class NormalizeBbox(object):
    def __call__(self,img,bbox):
        '''
        img: PIL image
        bbox: bbox(denormalized) in [x1,y1,x2,y2] format
              x1- horizontal comp and y1 - vertical comp of top left pixel
        '''
        assert isinstance(img, Image.Image) == True, "Image should be a PIL image" 
        w,h = img.size
        for i,b in enumerate(bbox):
            x1,y1,x2,y2,d = b
            bbox[i] = [x1/w, y1/h,x2/w,y2/h]
            bbox[i].append(d)
        return img, bbox

class CropBbox(object):
    def __call__(self,img,bbox):
        '''
        img: PIL image
        bbox: bbox(denormlized) in [x1,y1,x2,y2] format. 
              x1- horizontal comp and y1 - vertical comp of top left pixel
        '''
        assert len(bbox) == 1
        return img.crop(bbox[0][:4]), bbox

class Resize(transforms.Resize):
    def __call__(self,img,bbox):
        img = super().__call__(img)
        return img, bbox

class ToArray(object):
    def __call__(self,img,bbox):
        return np.array(img), np.array(bbox)

class ToTensor(transforms.ToTensor):
    def __call__(self,img,bbox):
        return super().__call__(img), torch.tensor(bbox)

class ConcatBbox(object):
    def __call__(self,img,bbox):
        assert len(bbox) ==1
        return np.concatenate((bbox[0],img),axis = None), bbox


