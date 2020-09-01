
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib
import pylab
import cv2
import os

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_ 0"] = "4"
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    dataDir='/tmp/pycharm_project_368'
    dataType='train2017'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    maskDir='/tmp/pycharm_project_368/masks/'

    coco=COCO(annFile)
    annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
    coco_kps=COCO(annFile)


    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print(len(nms))

    for i in range(len(nms)):
        catIds = coco.getCatIds(catNms=[nms[i]])
        imgIds = coco.getImgIds(catIds=catIds);
        for j in range(50, len(imgIds)):
            img = coco.loadImgs(imgIds[j])[0]
            print(img)
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            mask = coco.annToMask(anns[0])
            for k in range(len(anns)):
                maskfile_path = maskDir + img['file_name']
                maskfile_path = maskfile_path.replace(".jpg","")
                maskfile_path = maskfile_path + str(k) + '.png'
                mask = coco.annToMask(anns[k])
                print(mask)
                matplotlib.image.imsave(maskfile_path, mask)