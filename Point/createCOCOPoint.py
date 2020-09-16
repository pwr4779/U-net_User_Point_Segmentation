import errno
from random import randint

import matplotlib
import numpy as np
import os
import glob
import cv2

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
    # pointpatch = pointPatch
    temp = np.full(( 21 , 21 ), 255 )
    if(len(TPL)>0):
        index = randint(0,len(TPL)-1)
        kernel1d = cv2.getGaussianKernel(21, 5)
        kernel2d = np.outer(kernel1d, kernel1d.transpose())

        gaussianPath = temp * kernel2d
        k = 255 / gaussianPath[10][10]
        gaussianPath = k*gaussianPath
        patchY = TPL[index][0]
        patchX = TPL[index][1]
        for j in range(patchY-10, patchY+11):
            for k in range(patchX-10, patchX+11):
                if(j<0 or k<0 or j > y-1 or k>x-1):
                    continue
                patch[j][k] = gaussianPath[j-(patchY-10)][k-(patchX-10)]
    return patch

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_ 0"] = "4"
    for i in range(60):
        maskfile = '/Personmask2'
        pointfile = '/Person_Point2/point'+str(i)
        dirNames = '/tmp/pycharm_project_368'+maskfile
        fileNames = glob.glob(dirNames + "/*")
        print(len(fileNames))
        for i in range(len(fileNames)):
            print(str(i))
            image = cv2.imread(fileNames[i], cv2.IMREAD_GRAYSCALE)
            TPL = targetPixelList(image)
            patch = createPointLabel(TPL, image)
            dirpath = dirNames.replace(maskfile, pointfile)
            if not (os.path.isdir(dirpath)):
                os.makedirs(os.path.join(dirpath))
            path = fileNames[i].replace(maskfile, pointfile)
            cv2.imwrite(path, patch)

