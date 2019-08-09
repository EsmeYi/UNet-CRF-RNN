# U-Net with CRF-RNN layer in tensorflow and keras

This project aims at improving U-Net for medical images segmentation.
Our model was implemented using Tensorflow and Keras, and the CRF-RNN layer refers to this [repo](https://github.com/sadeepj/crfasrnn_keras/tree/master/src)

## Introducion

- U-Net with CRF-RNN layer paper:

1. [UNet-CRF-RNN]()

- Reference paper:

1. [U-Net](https://arxiv.org/pdf/1505.04597.pdf)
2. [FCN](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)
3. [CRF-RNN](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Conditional_Random_Fields_ICCV_2015_paper.pdf)

This repo provides an U-Net with the CRF-RNN layer, and also provides some extract models for comparison, like SegNet, FCN, vanilla U-Net and so on.

~~~
modelFns = {'unet':Models.VanillaUnet.VanillaUnet, 
            'segnet':Models.Segnet.Segnet , 
            'vgg_unet':Models.VGGUnet.VGGUnet , 
            'vgg_unet2':Models.VGGUnet.VGGUnet2 , 
            'fcn8':Models.FCN8.FCN8, 
            'fcn32':Models.FCN32.FCN32, 
            'crfunet':Models.CRFunet.CRFunet   }
~~~

## Usage

- data hierarchy 

~~~
    Use the Keras data generators to load train and test
    Image and label are in structure:
        train/
            img/
                0/
            gt/
                0/

        test/
            img/
                0/
            gt/
                0/

~~~
- Training parameters
~~~
'--batch_size', type=int, default=1, help='input batch size'
'--learning_rate', type=float, default=0.0001, help='learning rate'
'--lr_decay', type=float, default=0.9, help='learning rate decay'
'--epoch', type=int, default=80, help='# of epochs'
'--imSize', type=int, default=320, help='then crop to this size'
'--iter_epoch', type=int, default=0, help='# of iteration as an epoch'
'--num_class', type=int, default=2, help='# of classes'
'--checkpoint_path', type=str, default='', help='where checkpoint saved'
'--data_path', type=str, default='', help='where dataset saved. See loader.py to know how to organize the dataset folder'
'--load_from_checkpoint', type=str, default='', help='where checkpoint saved'
~~~

- Train your model
  ```
  python train.py --data_path ./datasets/ --checkpoint_path ./checkpoints/
  ``` 
- Visualize the train loss, dice score, learning rate, output mask, and first layer convolutional kernels per iteration in tensorboard
  ```
  tensorboard tensorboard --logdir=./checkpoints
  ``` 
- Evaluate your model
  ```
  python eval.py --data_path ./datasets/ --load_from_checkpoint ./checkpoints/model-xxxx
  ```
 
 ## Result
 
 - Dataset
 1. Hippocampus Segmentation: [ANDI](http://adni.loni.usc.edu) 
 2. Hippocampus Segmentation: [NITRC](https://www.nitrc.org/projects/hippseg_2011/)
 
 - Parameters
 
param | value
  --| -- 
batch_size | 5
epoch | 80
iter_epoch | 10
imSize | 320
learning_rate | 0.001
lr_decay	| 0.9
 
 - Result
 
 model | IU | DSC | PA
 --| -- | -- | --
CNN-CRF |	68.73%	| 73.22%	| 51.77%
FCN-8s |	59.61%	| 65.73%	| 44.26%
Segnet |	70.85%	| 79.01%	| 58.03%
Vanilla U-Net | 75.42%	| 83.49%	| 72.18%
U-Net-CRF | 78.00%	| 85.77%	| 79.05%
Our method	| 79.89%	| 87.31%	| 81.27%

