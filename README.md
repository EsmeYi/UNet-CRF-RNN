# Edge-aware FCN with CRF-RNN layer

change the list of your models in train.py and eval.py

~~~
modelFns = { 'unet':Models.VanillaUnet.VanillaUnet, 
            'segnet':Models.Segnet.Segnet , 
            'vgg_unet':Models.VGGUnet.VGGUnet , 
            'vgg_unet2':Models.VGGUnet.VGGUnet2 , 
            'fcn8':Models.FCN8.FCN8, 
            'fcn32':Models.FCN32.FCN32, 
            'crfunet':Models.CRFunet.CRFunet   }
~~~

data hierarchy 
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

### Usage

- Train your model
  ```bash
  python train.py --data_path ./datasets/ --checkpoint_path ./checkpoints/
  ``` 

- Visualize the train loss, dice score, learning rate, output mask, and first layer convolutional kernels per iteration in tensorboard
  ```bash
  tensorboard tensorboard --logdir=./checkpoints
  ``` 


- Evaluate your model
  ```bash
  python eval.py --data_path ./datasets/ --load_from_checkpoint ./checkpoints/model-0
  ```
