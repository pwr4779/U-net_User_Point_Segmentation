import errno
from random import randint

import matplotlib
import numpy as np
import os
import glob
import cv2
import pandas as pd
import PIL.Image as pilimg
from PIL import Image
import matplotlib.image as mpimg
temp = [[255.,255.,255.,255.,255.,255.,255.],
        [255.,255.,255.,255.,255.,255.,255.],
        [255.,255.,255.,255.,255.,255.,255.],
        [255.,255.,255.,255.,255.,255.,255.],
        [255.,255.,255.,255.,255.,255.,255.],
        [255.,255.,255.,255.,255.,255.,255.],
        [255.,255.,255.,255.,255.,255.,255.]]

def targetPixelList(image):
    x = len(image[0])
    y = len(image)
    pointlist = []
    for i in range(y):
        for j in range(x):
            if(image[i][j] > 100):
                pointlist.append([i,j])
    return pointlist


def createPointLabel(TPL, image):
    y = len(image)
    x = len(image[0])
    patch = np.zeros((y,x))

    if(len(TPL)>0):
        index = randint(0,len(TPL)-1)
        kernel1d = cv2.getGaussianKernel(7, 2.5)
        kernel2d = np.outer(kernel1d, kernel1d.transpose())
        gaussianPath = kernel2d * temp
        patchY = TPL[index][0]
        patchX = TPL[index][1]
        for j in range(patchY-3, patchY+4):
            for k in range(patchX-3, patchX+4):
                if(j<0 or k<0 or j > y-1 or k>x-1):
                    continue
                patch[j][k] = gaussianPath[j-(patchY-3)][k-(patchX-3)]
    return patch

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_ 0"] = "4"

    dirNames = '/tmp/pycharm_project_368/masks'
    # #print(dirNames)
    # #for i in range():
    # if not (os.path.isdir('/tmp/pycharm_project_368/DAVIS/Annotations/Point')):
    #     os.makedirs(os.path.join('/tmp/pycharm_project_368/DAVIS/Annotations/Point'))
    # for i in range(len(dirNames)):
    fileNames = glob.glob(dirNames + "/*")
    for file in fileNames:
        src = cv2.imread(file, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        TPL = targetPixelList(image)
        patch = createPointLabel(TPL, image)
        dirpath = dirNames.replace('/masks', '/points')
        if not (os.path.isdir(dirpath)):
            os.makedirs(os.path.join(dirpath))
        path = file.replace('/masks', '/points')
        cv2.imwrite(path, patch)

