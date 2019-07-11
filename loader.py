"""
Utility of loading data, processing and deprocessing 
"""
from data_generator.image import ImageDataGenerator, img_to_array
import scipy.misc as misc
import numpy as np
import os, glob, itertools
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


# data normalization 
def preprocess(img, label, label_classes, mean, std):

    out_img = preprocess_img(img, mean, std)

    label_mclass = preprocess_label(label, label_classes)

    # label_mclass.shape, out_img.shape: (256, 256, 4) (1, 256, 256, 3) 
    return out_img, label_mclass.astype(np.int32)

def preprocess_label(label, label_classes):
    if len(label.shape) == 4:
        label = label[:,:,:,0]

    num_class = label_classes.size
    # if num_class == 2:
    #     label = label / label.max() # if the loaded label is binary has only [0,255], then we normalize it
    # if num_class > 2 :
    batch_num, ny, nx = label.shape[0], label.shape[1], label.shape[2]
    label_mclass = np.zeros((batch_num, ny, nx, num_class), dtype=np.float32)
    for c in range(num_class):
        label_mclass[:, : , :, c ] = (label == label_classes[c]).astype(int)
    return label_mclass

def preprocess_img(img, mean, std):
    out_img = img / img.max() # scale to [0,1]
    out_img = (out_img - np.array(mean).reshape(1,1,3)) / np.array(std).reshape(1,1,3)
    return out_img

def deprocess(img, label, pred_map, label_classes, mean, std):
    out_img = deprocess_img(img, std, mean)
    label = deprocess_label(label, label_classes)
    pred_map = deprocess_pred(pred_map, label_classes)
    return out_img.astype(np.uint8), label.astype(np.uint8), pred_map.astype(np.uint8)

def deprocess_img(img, std, mean):
    out_img = img / img.max() # scale to [0,1]
    out_img = (out_img * np.array(std).reshape(1,1,3)) + np.array(mean).reshape(1,1,3) 
    out_img = out_img * 255.0
    return out_img

def deprocess_label(label,label_classes):
    num_class = label.shape[2]
    # if num_class == 2:
    #     label = label * 255.0
    # if num_class > 2 :
    ny, nx = label.shape[0], label.shape[1]
    label_single = np.zeros((ny, nx), dtype=np.float32)
    for c in range(num_class):
        current_ch = label[:,:,c]
        current_ch[current_ch != 0] = label_classes[c] 
        label_single[:,:] += current_ch
    return label_single

def deprocess_pred(pred,label_classes):
    num_class = label_classes.size
    if num_class == 2:
        pred = pred * 255.0
    if num_class > 2 :
        for c in range(num_class):
            pred[pred==c] = label_classes[c]
    return pred

'''
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

'''
def dataLoader(path, batch_size, imSize, label_classes, train_mode=True, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
    # image normalization default: scale to [-1,1]
    def imerge(a, b):
        for img, label in itertools.izip_longest(a,b):
            # j is the mask: 1) gray-scale and int8     
            # img.shape: (1, 256, 256, 3)
            img, label = preprocess(img, label, label_classes, mean, std)
            yield img, label
    # augmentation parms for the train generator
    if train_mode:
        train_data_gen_args = dict(
                        horizontal_flip=True,
                        vertical_flip=True,
                        )
    else:
        train_data_gen_args = dict()
    
    # seed has to been set to synchronize img and mask generators
    seed = 1
    train_image_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
                                path+'img',
                                class_mode=None,
                                target_size=imSize,
                                batch_size=batch_size,
                                seed=seed,
                                shuffle=train_mode)
    train_mask_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
                                path+'gt',
                                class_mode=None,
                                target_size=imSize,
                                batch_size=batch_size,
                                color_mode='grayscale',
                                seed=seed,
                                shuffle=train_mode)
                                
    samples = train_image_datagen.samples
    generator = imerge(train_image_datagen, train_mask_datagen)
    return generator, samples

def testDataLoader(path, imSize, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
    # image normalization default: scale to [-1,1]
    def imerge(a):
        for image in itertools.cycle(a):
            # j is the mask: 1) gray-scale and int8 
            img = image[0]
            fname = image[1]
            img = preprocess_img(img, mean, std)
            image = []
            image.append(img)
            image.append(fname)
            yield image 

    data_gen_args = dict()
    image_datagen = ImageDataGenerator(**data_gen_args).flow_from_directory(
                                path+'img',
                                class_mode=None,
                                target_size=imSize,
                                batch_size=1,
                                seed=1,
                                shuffle=False)
                                
    samples = image_datagen.samples
    generator = imerge(image_datagen)
    return generator, samples

def testDataLoader2(path, imSize, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
    data_path = path + 'img'
    data = []
    cubes_name = []
    i = 0
    for cube_name in os.listdir(data_path):
        cube_path = data_path + '/' + cube_name
        one_cube = []
        for img_name in os.listdir(cube_path):
            i += 1
            img_path = cube_path + '/' + img_name
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            hw_tuple = (imSize[1], imSize[0])
            if img.size != hw_tuple:
                img = img.resize(hw_tuple)
            x = img_to_array(img, 'channels_last')
            img = np.array(img)
            h,w,c = img.shape[0], img.shape[1], img.shape[2]
            img = img.reshape(1,h,w,c)
            img = preprocess_img(img, mean, std)
            if i ==1:

                im = img.reshape(h,w,c)
                im = deprocess_img(im, std, mean)
                im = Image.fromarray(im.astype(np.uint8), mode='RGB')
                im.save('check.png')
            
            one_cube.append(img)
        data.append(one_cube)
        cubes_name.append(cube_name)
    return np.array(data), np.array(cubes_name)

