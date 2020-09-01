import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import load_model

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="4"
    pointfiles = glob.glob('/tmp/pycharm_project_368/points/*')
    length = int(len(pointfiles)/20)

    #Train
    trainMask = []
    trainImg = []
    trainPoint = []

    # valid
    validMask = []
    validImg = []
    validPoint = []

    m = int(length * 8 / 10)
    for i in range(0, m):
        # point가 있다면 mask랑 이미지 넣음.
        trainPoint.append(pointfiles[i])
        base = os.path.basename(pointfiles[i])
        filename=os.path.splitext(base)[0]
        Imgpath = '/tmp/pycharm_project_368/train2017/' + filename[0:12] +'.jpg'
        if(not os.path.isfile(Imgpath)):
            continue
        trainImg.append(Imgpath)
        maskpath = pointfiles[i].replace('/points', '/masks')
        trainMask.append(maskpath)
        # print(pointfiles[i])
        # print(Imgpath)
        # print(maskpath)
    for i in range(m, length):
        validPoint.append(pointfiles[i])
        Imgpath = '/tmp/pycharm_project_368/train2017/' + filename[0:12] +'.png'
        validImg.append(Imgpath)
        maskpath = pointfiles[i].replace('/points', '/masks')
        validMask.append(maskpath)
        # print(pointfiles[i])
        # print(Imgpath)
        # print(maskpath)

    train_x = np.zeros((len(trainImg),288,480, 4))
    valid_x = np.zeros((len(validImg),288,480, 4))
    train_mask_y = np.zeros((len(trainMask),288,480))
    valid_mask_y = np.zeros((len(validMask),288,480))

    #입력(이미지,Point정보) 288*480 형태로 resize 후 4차원으로 만들기
    #train
    for index in range(len(trainImg)):
        mask = cv2.imread(trainMask[index], cv2.IMREAD_GRAYSCALE)
        dstmask = cv2.resize(mask, dsize=(480, 288), interpolation=cv2.INTER_AREA)
        img = cv2.imread(trainImg[index], cv2.IMREAD_COLOR)
        dstimg = cv2.resize(img, dsize=(480, 288), interpolation=cv2.INTER_AREA)
        Point = cv2.imread(trainPoint[index], cv2.IMREAD_GRAYSCALE)
        dstPoint = cv2.resize(Point, dsize=(480, 288), interpolation=cv2.INTER_AREA)
        tmask = np.array(dstmask)
        timg = np.array(dstimg)
        train_Point = np.array(dstPoint)
        train_x[index,:,:,0:3] = timg
        train_x[index,:,:,3] = train_Point
        train_mask_y[index,:,:] = tmask

    #valid
    for index in range(len(validImg)):
        mask = cv2.imread(validMask[index], cv2.IMREAD_GRAYSCALE)
        dstmask = cv2.resize(mask, dsize=(480, 288), interpolation=cv2.INTER_AREA)
        img = cv2.imread(validImg[index], cv2.IMREAD_COLOR)
        dstimg = cv2.resize(img, dsize=(480, 288), interpolation=cv2.INTER_AREA)
        Point = cv2.imread(validPoint[index], cv2.IMREAD_GRAYSCALE)
        dstPoint = cv2.resize(Point, dsize=(480, 288), interpolation=cv2.INTER_AREA)
        vmask = np.array(dstmask)
        vimg = np.array(dstimg)
        valid_Point =np.array(dstPoint)
        valid_x[index, :, :, 0:3] = vimg
        valid_x[index, :, :, 3] = valid_Point
        valid_mask_y[index,:,:] = vmask

    train_x = train_x.reshape(len(train_x), 288, 480, 4).astype('float32')/255.0
    valid_x = valid_x.reshape(len(valid_x), 288, 480, 4).astype('float32')/255.0
    train_mask_y = train_mask_y.reshape(len(train_mask_y), 288, 480, 1).astype('float32')/255.0
    valid_mask_y = valid_mask_y.reshape(len(valid_mask_y), 288, 480, 1).astype('float32')/255.0
    print('start')
    model = load_model('UNetv8.h5')
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(patience=20)
    results = model.fit(train_x, train_mask_y, validation_data=(valid_x, valid_mask_y), epochs=150, batch_size=16, verbose=1, callbacks=[early_stopping])
    model.save("UNetv9.h5")
    print('finish')