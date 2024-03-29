# keras 实现的fcn

import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

#############################data process################################
dir_data = "H:/data/dataset1"
dir_seg = dir_data + "/annotations_prepped_train/"
dir_img = dir_data + "/images_prepped_train/"

sns.set_style("whitegrid", {'axes.grid' : False})

ldseg = np.array(os.listdir(dir_seg))#获取文件名称,保存在numpy数组中

fnm = ldseg[0]
seg = cv2.imread(dir_seg + fnm)
img_is = cv2.imread(dir_img + fnm )
mi, ma = np.min(seg), np.max(seg)
n_classes = ma - mi + 1

#通过第一个样本了解数据
def data_detail():
    fnm = ldseg[0]
    seg = cv2.imread(dir_seg + fnm)
    img_is = cv2.imread(dir_img + fnm )
    print(fnm)
    print("seg.shape={}, img_is.shape={}".format(seg.shape,img_is.shape))
    mi, ma = np.min(seg), np.max(seg)
    n_classes = ma - mi + 1
    print("minimum seg = {}, maximum seg = {}, Total number of segmentation classes = {}".format(mi, ma, n_classes))
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(img_is)
    ax.set_title("original image")
    plt.show()
    fig = plt.figure(figsize=(15,10))
    for k in range(mi, ma+1):
        ax = fig.add_subplot(3, n_classes/3, k+1)
        ax.imshow((seg == k)*1.0)
        ax.set_title("label = {}".format(k))
    plt.show()

def give_color_to_seg_img(seg, n_classes):
    '''
    seg:(width, height, 3)
    '''
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3)).astype('float')
    colors = sns.color_palette("hls", n_classes)
    for c in range(n_classes):
        segc = (seg==c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))
    return seg_img

input_height , input_width = 224 , 224
output_height , output_width = 224 , 224

# for fnm in ldseg[np.random.choice(len(ldseg),4,replace=False)]:
#     fnm = fnm.split(".")[0]
#     seg = cv2.imread(dir_seg + fnm + ".png") # (360, 480, 3)
#     img_is = cv2.imread(dir_img + fnm + ".png")
#     seg_img = give_color_to_seg_img(seg,n_classes)

#     fig = plt.figure(figsize=(20,40))
#     ax = fig.add_subplot(1,4,1)
#     ax.imshow(seg_img)
    
#     ax = fig.add_subplot(1,4,2)
#     ax.imshow(img_is/255.0)
#     ax.set_title("original image {}".format(img_is.shape[:2]))
    
#     ax = fig.add_subplot(1,4,3)
#     ax.imshow(cv2.resize(seg_img,(input_height , input_width)))
    
#     ax = fig.add_subplot(1,4,4)
#     ax.imshow(cv2.resize(img_is,(output_height , output_width))/255.0)
#     ax.set_title("resized to {}".format((output_height , output_width)))
#     plt.show()

def getImageArr(path, width, height):
    img = cv2.imread(path, 1)
    img  =np.float32(cv2.resize(img, (width, height)))/127.5-1
    return img

def getSegmentationArr( path , nClasses ,  width , height  ):
    seg_labels = np.zeros((  height , width  , nClasses ))
    img = cv2.imread(path, 1)
    img = cv2.resize(img, ( width , height ))
    img = img[:, : , 0]

    for c in range(nClasses):
        seg_labels[: , : , c ] = (img == c ).astype(int)
    ##seg_labels = np.reshape(seg_labels, ( width*height,nClasses  ))
    return seg_labels

images = os.listdir(dir_img)
images.sort()
segmentations  = os.listdir(dir_seg)
segmentations.sort()

X = []
Y = []
for im , seg in zip(images,segmentations) :
    X.append( getImageArr(dir_img + im , input_width , input_height )  )
    Y.append( getSegmentationArr( dir_seg + seg , n_classes , output_width , output_height )  )

X, Y = np.array(X) , np.array(Y)
print(X.shape,Y.shape)

from sklearn.utils import shuffle
train_rate = 0.85
index_train = np.random.choice(X.shape[0],int(X.shape[0]*train_rate),replace=False)
index_test = list(set(range(X.shape[0]))-set(index_train))
X, Y = shuffle(X, Y)
X_train, y_train = X[index_train], Y[index_train]
X_test, y_test = X[index_test], Y[index_test]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

################################data process##########################################

################################fcn models###########################################

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
import pandas as pd
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.visible_device_list = "0" 
set_session(tf.Session(config=config))   

print("python {}".format(sys.version))
print("keras version {}".format(keras.__version__)); del keras
print("tensorflow version {}".format(tf.__version__))

# location of VGG weights
VGG_Weights_path = "H:/data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

def FCN8(nClasses, input_height=224, input_width=224):
    assert input_height%32 == 0
    assert input_width%32 == 0
    IMAGE_ORDERING =  "channels_last" 

    img_input = Input(shape=(input_height, input_width, 3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    f1 = x

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
    f2 = x

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    pool3 = x

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)## (None, 14, 14, 512) 

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)## (None, 7, 7, 512)

    vgg  = Model(img_input, pool5)
    vgg.load_weights(VGG_Weights_path)

    n = 4096
    o = ( Conv2D( n , ( 7 , 7 ) , activation='relu' , padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = ( Conv2D( n , ( 1 , 1 ) , activation='relu' , padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)

    conv7_4 = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(4,4) , use_bias=False, data_format=IMAGE_ORDERING )(conv7)
    pool411 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(pool4)
    pool411_2 = (Conv2DTranspose( nClasses , kernel_size=(2,2) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING ))(pool411)
    
    pool311 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)
        
    o = Add(name="add")([pool411_2, pool311, conv7_4 ])
    o = Conv2DTranspose( nClasses , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False, data_format=IMAGE_ORDERING )(o)
    o = (Activation('softmax'))(o)

    model = Model(img_input, o)
    return model

model = FCN8(nClasses=n_classes, input_height=224, input_width=224)
model.summary()
################################fcn models###########################################
#训练模型
# from keras import optimizers
# sgd = optimizers.SGD(lr=1e-2, decay=5**(-4), momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# hist1 = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=200, verbose=2)

#保存模型
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_dataset1_fcn.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
# model.save(model_path)

#保存history到json文件中
import json
his_name = 'keras_dataset1_fcn_history.json'
his_path = os.path.join(save_dir, his_name)
# with open(his_path, 'w') as f:
#     json.dump(hist1.history, f)

#从保存的history中绘制图像
with open(his_path, 'r') as f:
    history = json.load(f)
for key in ['loss', 'val_loss', 'acc', 'val_acc']:
    plt.plot(history[key],label=key)
plt.legend()
plt.show()

#加载保存的模型进行预测
model.load_weights(model_path)
y_pred = model.predict(X_test)
y_predi = np.argmax(y_pred, axis=3)
y_testi = np.argmax(y_test, axis=3)
print(y_testi.shape,y_predi.shape)

#对每个类计算IoU
def IOU(Yi, y_predi):
    IoUs = []
    Nclass = int(np.max(Yi)) + 1
    for c in range(Nclass):
        TP = np.sum( (Yi == c)&(y_predi==c) )
        FP = np.sum( (Yi != c)&(y_predi==c) )
        FN = np.sum( (Yi == c)&(y_predi != c)) 
        IoU = TP/float(TP + FP + FN)
        print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(c,TP,FP,FN,IoU))
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    print("_________________")
    print("Mean IoU: {:4.3f}".format(mIoU))

IOU(y_testi, y_predi)

#显示预测的结果
shape = (224,224)
n_classes= 12
for i in range(10):
    img_is  = (X_test[i] + 1)*(255.0/2)
    seg = y_predi[i]
    segtest = y_testi[i]

    fig = plt.figure(figsize=(10,30))    
    ax = fig.add_subplot(1,3,1)
    ax.imshow(img_is/255.0)
    ax.set_title("original")
    
    ax = fig.add_subplot(1,3,2)
    ax.imshow(give_color_to_seg_img(seg,n_classes))
    ax.set_title("predicted class")
    
    ax = fig.add_subplot(1,3,3)
    ax.imshow(give_color_to_seg_img(segtest,n_classes))
    ax.set_title("true class")
    plt.show()