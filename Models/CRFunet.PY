import os 
import numpy as np
import cv2
import tensorflow as tf
from keras.models import *
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, ZeroPadding2D
from keras import backend as keras
from keras import layers
import sys
sys.path.insert(1, './crf_as_rnn')
from crfrnn_layer import CrfRnnLayer
from utils import dice_coef_2, mean_IU


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

def create_model(img_shape, num_class):

    concat_axis = 3
    # input
    inputs = Input(shape = img_shape)

    # Unet convolution block 1
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(inputs)
    print "conv1 shape:",conv1.shape
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv1)
    print "conv1 shape:",conv1.shape        
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print "pool1 shape:",pool1.shape

    # Unet convolution block 2
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool1)
    print "conv2 shape:",conv2.shape
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv2)
    print "conv2 shape:",conv2.shape
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print "pool2 shape:",pool2.shape

    # Unet convolution block 3
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool2)
    print "conv3 shape:",conv3.shape
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv3)
    print "conv3 shape:",conv3.shape
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print "pool3 shape:",pool3.shape

    # Unet convolution block 4
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool3)
    print "conv4 shape:",conv4.shape
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv4)
    print "conv4 shape:",conv4.shape
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    print "pool4 shape:",pool4.shape

    # Unet convolution block 5
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool4)
    print "conv5 shape:",conv5.shape
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv5)
    print "conv5 shape:",conv5.shape
    # drop5 = Dropout(0.5)(conv5)

    # Unet up-sampling block 1; Concatenation with crop_conv4
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    print "up6 shape:",up6.shape
    ch, cw = get_crop_shape(conv4, up6)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    print "crop_conv4 shape:",crop_conv4.shape
    merge6 = concatenate([crop_conv4,up6], axis = 3)
    print "merge6 shape:",merge6.shape
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    print "conv6 shape:",conv6.shape
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    print "conv6 shape:",conv6.shape
    
    # Unet up-sampling block 2; Concatenation with crop_conv3
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    print "up7 shape:",up7.shape
    ch, cw = get_crop_shape(conv3, up7)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    print "crop_conv3 shape:",crop_conv3.shape
    merge7 = concatenate([crop_conv3,up7], axis = 3)
    print "merge7 shape:",merge7.shape
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    print "conv7 shape:",conv7.shape
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    print "conv7 shape:",conv7.shape
   
    # Unet up-sampling block 3; Concatenation with crop_conv2
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    print "up8 shape:",up8.shape
    ch, cw = get_crop_shape(conv2, up8)
    crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
    print "crop_conv2 shape:",crop_conv2.shape
    merge8 = concatenate([crop_conv2,up8], axis = 3)
    print "merge8 shape:",merge8.shape
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    print "conv8 shape:",conv8.shape
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    print "conv8 shape:",conv8.shape

    # Unet up-sampling block 4; Concatenation with crop_conv1
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    print "up9 shape:",up9.shape
    ch, cw = get_crop_shape(conv1, up9)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    print "crop_conv1 shape:",crop_conv2.shape
    merge9 = concatenate([crop_conv1,up9], axis = 3)
    print "merge9 shape:",merge9.shape
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    print "conv9 shape:",conv9.shape
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    print "conv9 shape:",conv9.shape

    conv9 = Conv2D(num_class, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    print "conv9 shape:",conv9.shape
    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
    print "conv9 shape:",conv9.shape

    # conv10 = Conv2D(num_class, (1, 1))(conv9)
    # print "conv10 shape:",conv10.shape

    # Add Crf_rnn_layer
    output = CrfRnnLayer(image_dims=img_shape,
                         num_classes=num_class,
                         theta_alpha=160.,
                         theta_beta=3.,
                         theta_gamma=3.,
                         num_iterations=10,
                         name='crfrnn')([conv9, inputs])
    print "output shape", output.shape


    model = Model(input = inputs, output = output)

    return model

def get_cost(label, pred, cost_name, weight_map = None, a = 1.0):
    """
    Constructs the cost function, either cross_entropy, weighted_cross_entropy or dice_coefficient.
    """

    def dice_coef(y_true, y_pred, smooth=1.):
        intersection = K.sum(y_true * y_pred)
        union = K.sum(y_true) + K.sum(y_pred)
        return  1-K.mean( (2. * intersection + smooth) / (union + smooth))

    if cost_name == "log_loss":
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=pred))
                
    elif cost_name == "dice_loss":
        label = label[0,:,:,1]
        pred = pred[0,:,:,1]
        dice =  dice_coef(label, pred)
        loss = 1-dice
        
    elif cost_name == "log_dice_loss":
        ##loss = log_loss + 0.01*dice_loss overall mean IU: 0.671168076957 
        log_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=pred))
        dice_loss = tf.reduce_mean(dice_coef_2(label, pred))
        loss = log_loss + 0.01*dice_loss

    elif cost_name == "staged_loss":
        log_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=pred))
        d_loss = dice_coef(label, pred)
        loss = a*log_loss+(1-a)*d_loss*0.1

    elif cost_name == "weighted_loss":
        loss_map = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=pred))
        if weight_map == None:
            raise ValueError("Weight map is none")
        weighted_loss = tf.multiply(loss_map, weight_map)
        loss = tf.reduce_mean(weighted_loss) 
    else:
        raise ValueError("Unknown cost function: "%cost_name)
        
    return loss

def define_map(img):
    kernel_sharpen_1 = np.array([
                            [-1,-1,-1],
                            [-1,9,-1],
                            [-1,-1,-1]])

    img = img[:,:,0]
    img = np.uint8(img)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.filter2D(img,-1,kernel_sharpen_1)
    edges = cv2.Canny(img,0,1)
    # edges = cv2.resize(edges,(512, 512))
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # corners = cv2.goodFeaturesToTrack(edges,7,0.001,10)
    # corners = cv2.cornerSubPix(edges, corners, (5, 5), (-1, -1), criteria) 
    # corners = np.int0(corners)
    # print len(corners)
    w_map = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
    edges = np.argwhere(edges == 255)
    for i in edges:
        x,y = i.ravel()
        w_map[x,y] = 5
    # cv2.imshow("image",w_map)
    # cv2.waitKey(0)
    # w_map = tf.reduce_sum(w_map, axis=1)
    return w_map

if __name__ == '__main__':
    # img1 = np.asarray(cv2.imread('./palette.png', 0)).astype('int8').astype('float32')
    img1 = cv2.imread('./palette.png', 0)
    print define_map(img1)
    # img2 = np.asarray(cv2.imread('./0592.png', 0)).astype('int8').astype('float32')
    # loss = dice_coef(img1,img2)
    # sess = tf.Session()
    # print sess.run(loss)

