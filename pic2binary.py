# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import PIL.Image as Image
import pickle as p
import os
import matplotlib.pyplot as pyplot

classDict = {'person':1,
             'bird':2,
             'cat':3,
             'cow':4,
             'dog':5,
             'horse':6,
             'sheep':7,
             'aeroplane':8,
             'bicycle':9,
             'boat':10,
             'bus':11,
             'car':12,
             'motorbike':13,
             'train':14,
             'bottle':15,
             'chair':16,
             'dining':17,
             'table':18,
             'potted':19,
             'plant':20,
             'sofa':21,
             'tvmonitor':22,
             'pottedplant':23,
             'diningtable':24
             }
y_list = np.zeros([1,24])
def image_to_array(filenames,width=0,depth=0,className='person'):
    storePath = "/home/jackey/private/project/ML/data/VOC2012/Binary/test2.bin"
    image = Image.open(filenames)
    pad_w = 500 - width
    pad_d = 500 - depth
    r,g,b = image.split()
    r_arr = np.array(r).reshape([width,depth])
    r_arr =np.pad(r_arr,((0,pad_w),(0,pad_d)),'constant')
    r_arr = np.array(r_arr).reshape([1,500,500,1])
    g_arr = np.array(g).reshape([width,depth])
    g_arr =np.pad(g_arr,((0,pad_w),(0,pad_d)),'constant')
    g_arr = np.array(g_arr).reshape([1,500,500,1])
    b_arr = np.array(b).reshape([width,depth])
    b_arr =np.pad(b_arr,((0,pad_w),(0,pad_d)),'constant')
    b_arr = np.array(b_arr).reshape([1,500,500,1])
    image_arr = np.concatenate((r_arr, g_arr, b_arr),axis=3)
    y  = tagNUM(className)
    probability_list = np.zeros([1,24])
    probability_list[0,y-1]=1
    """
    with open(storePath, mode='wa') as f:
        p.dump(image_arr, f)
    """
    print('jrep 2 array success')
    return image_arr,probability_list

def bin2tensor2(fileName,isTag):
    arr = np.fromfile(fileName,dtype=np.float32)
    if isTag:
        print('its tag')
    else:
        arr = np.reshape(arr,([-1,500,500,3]))
        print(arr.shape)
    return arr

def tagNUM(tagName):
    #classDict['person']
    #print(classDict.get(tagName))
    return classDict.get(tagName)
"""
xs,ys =  image_to_array("/home/jackey/private/project/ML/data/VOC2012/JPEGImagesTmp/2009_001000.jpg",500,121)
xs,ys =  image_to_array("/home/jackey/private/project/ML/data/VOC2012/JPEGImagesTmp/2009_000001.jpg",360,270)
bin2tensor2('/home/jackey/private/project/ML/data/VOC2012/Binary/input_tag.txt',isTag=True)
bin2tensor2('/home/jackey/private/project/ML/data/VOC2012/Binary/input_x.txt',isTag=False)
"""

#print(xs.shape)
#print(ys)
