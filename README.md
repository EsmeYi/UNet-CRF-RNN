# Edge-aware U-Net with CRF-RNN layer

This project aims at improving U-Net for medical images segmentation.

## Introducion

- The paper of Edge-aware U-Net with CRF-RNN layer

0. [UNet-CRF-RNN]()

- Reference paper:

1. [U-Net]()
2. [FCN]()
3. [CRF-RNN]()


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
