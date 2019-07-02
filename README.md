# Edge-aware U-Net with CRF-RNN layer

This project aims at improving U-Net for medical images segmentation.

## Introducion

- The paper of Edge-aware U-Net with CRF-RNN layer

0. [UNet-CRF-RNN]()

- Reference paper:

1. [U-Net](https://arxiv.org/pdf/1505.04597.pdf)
2. [FCN](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)
3. [CRF-RNN](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Conditional_Random_Fields_ICCV_2015_paper.pdf)


This repo provides an edge-aware U-Net with CRF-RNN layer, and also provides some extract models for comparison, like SegNet, FCN, vanilla U-Net and so on.

~~~
modelFns = { 'unet':Models.VanillaUnet.VanillaUnet, 
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
  python eval.py --data_path ./datasets/ --load_from_checkpoint ./checkpoints/model-0
  ```
 
 ## Result
 
 model | IU | DSC | PA
 --| -- | -- | --
CNN-CRF |	68.73%	| 73.22%	| 51.77%
FCN-8s |	59.61%	| 65.73%	| 44.26%
Segnet |	70.85%	| 79.01%	| 58.03%
Vanilla U-Net | 75.42%	| 83.49%	| 72.18%
U-Net-CRF | 78.00%	| 85.77%	| 79.05%
Our method	| 79.89%	| 87.31%	| 81.27%

