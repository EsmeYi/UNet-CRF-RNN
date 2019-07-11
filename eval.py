"""
Models evaluation and performance comparison
Models: Segnet, Vgg_Unet, FCN, Vanilla Unet, CRF_Unet
Evaluation metrics: DICE, IoU, meanPixleAccuracy
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import Models
import os, sys
import numpy as np
import scipy.misc as misc
from utils import dice_coef, dice_coef_loss
from loader import dataLoader, deprocess
from PIL import Image
from utils import VIS, mean_IU

# configure args
from opts import *
from opts import dataset_mean, dataset_std # set them in opts

modelFns = { 'unet':Models.VanillaUnet.VanillaUnet, 
			'segnet':Models.Segnet.Segnet , 
			'vgg_unet':Models.VGGUnet.VGGUnet , 
			'vgg_unet2':Models.VGGUnet.VGGUnet2 , 
			'fcn8':Models.FCN8.FCN8, 
			'fcn32':Models.FCN32.FCN32, 
			'crfunet':Models.CRFunet.CRFunet   }

vis = VIS(save_path=opt.load_from_checkpoint)

# configuration session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# define data loader
img_shape = [opt.imSize, opt.imSize]
label_classes = vis.palette_info()
test_generator, test_samples = dataLoader(opt.data_path+'/val/', 1,  img_shape, label_classes, train_mode=False)
# define model, the last dimension is the channel
label = tf.placeholder(tf.float32, shape=[None]+img_shape+[len(label_classes)])
# define model
with tf.name_scope('network'):
    modelFN = modelFns[ "crfunet" ]
    model = modelFN(opt.num_class, img_shape=img_shape+[3])
    img = model.input
    pred = model.output
# define loss
with tf.name_scope('cross_entropy'): 
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=pred))

saver = tf.train.Saver() # must be added in the end

''' Main '''
init_op = tf.global_variables_initializer()
sess.run(init_op)
with sess.as_default():
    # restore from a checkpoint if exists
    try:
        saver.restore(sess, opt.load_from_checkpoint)
        print ('--> load from checkpoint '+opt.load_from_checkpoint)
    except:
        print ('unable to load checkpoint ...')
        sys.exit(0)
    dice_score = 0
    for it in range(0, test_samples):
        x_batch, y_batch = next(test_generator)
        # tensorflow wants a different tensor order
        print (x_batch.shape)
        feed_dict = {   
                        img: x_batch,
                        label: y_batch
                    }
        loss, pred_logits = sess.run([cross_entropy_loss, pred], feed_dict=feed_dict)
        # pred_logits.shape: (1, 256, 256, 4)
        pred_map = np.argmax(pred_logits[0], axis=2)
        # pred_map.shape: (256, 256)
        mean_iou, pixel_acc, dice = vis.add_sample(pred_map, y_batch[0])
        
        im, gt, gt_pred = deprocess(x_batch[0], y_batch[0], pred_map, label_classes, dataset_mean, dataset_std )
        vis.save_seg(gt_pred, name='{0:}_{1:.3f}.png'.format(it, mean_iou), im=im, gt=gt)

        print ('[iter %f]: loss=%f, meanIU=%f, PixelAccuracy=%f, Dice=%f' % (it, loss, mean_iou, pixel_acc, dice))

    vis.compute_scores()
