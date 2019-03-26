from keras import backend as K
import tensorflow as tf
import numpy as np
from PIL import Image
import os, copy, shutil, json
from skimage import measure

class VIS:
    def __init__(self, save_path):

        self.path = save_path
        # TODO
        self.semantic_label = None

        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
        os.mkdir(self.path)

        self.mean_iu = []
        self.cls_iu = []
        self.pix_acc = []
        self.mean_dice = []
        self.cls_dice = []
        self.score_history = {}
        self.suffix = str(np.random.randint(1000))

        self.exam_img = Image.open('palette.png')

        self.palette = self.exam_img.palette
        # self.orginal_size = (self.exam_img.shape[1], self.exam_img.shape[0])

    def palette_info(self):
        return np.unique(self.exam_img)

    def save_seg(self, label_im, name, im=None, gt=None):
        seg = Image.fromarray(label_im.astype(np.uint8), mode='P') # must convert to int8 first
        seg.palette = copy.copy(self.palette)
        if gt is not None or im is not None:
            gt = Image.fromarray(gt.astype(np.uint8), mode='P') # must convert to int8 first]
            gt.palette = copy.copy(self.palette)
            im = Image.fromarray(im.astype(np.uint8), mode='RGB')
            I = Image.new('RGB', (label_im.shape[1]*3, label_im.shape[0]))
            I.paste(im,(0,0))
            I.paste(gt,(320,0))
            I.paste(seg,(640,0))
            I.save(os.path.join(self.path, name))
        else:   
            seg.save(os.path.join(self.path, name))   

    def save_seg2(self, label_im, name, im=None):
        seg = Image.fromarray(label_im.astype(np.uint8), mode='P') # must convert to int8 first
        seg.palette = copy.copy(self.palette)
        if im is not None:
            im = Image.fromarray(im.astype(np.uint8), mode='RGB')
            I = Image.new('RGB', (label_im.shape[1]*2, label_im.shape[0]))
            I.paste(im,(0,0))
            I.paste(seg,(256,0))
            if '/' in name:
                parent_dir = name.split('/')[0]
                fname = name.split('/')[1]
                dir_name = os.path.join(self.path, parent_dir)
                if os.path.exists(dir_name) == False:
                    os.mkdir(os.path.join(self.path, parent_dir))
            I.save(os.path.join(self.path, name))
        else:   
            seg.save(os.path.join(self.path, name))  

    def add_sample(self, pred, gt):
        score_mean, score_cls = mean_IU(pred, gt)
        p_accuracy = pixel_accuracy(pred, gt)
        dice_mean, dice_cls = dice_coef_2(pred, gt)
        self.mean_iu.append(score_mean)
        self.cls_iu.append(score_cls)
        self.pix_acc.append(p_accuracy)
        self.mean_dice.append(dice_mean)
        self.cls_dice.append(dice_cls)

        return score_mean, p_accuracy, dice_mean


    def compute_scores(self, suffix=0):
        meanIU = np.mean(np.array(self.mean_iu))
        meanIU_per_cls = np.mean(np.array(self.cls_iu), axis=0)
        mean_pix_acc = np.mean(np.array(self.pix_acc))
        mean_dice = np.mean(np.array(self.mean_dice))
        meanDice_per_cls = np.mean(np.array(self.cls_dice), axis=0)
        print ('-'*20)
        print ('overall mean IU: {} '.format(meanIU))
        print ('overall mean Pixel Accuracy: {} '.format(mean_pix_acc))
        print ('overall mean Dice: {} '.format(mean_dice))
        print ('mean IU per class')
        for i, c in enumerate(meanIU_per_cls):
            print ('\t class {}: {}'.format(i,c))
        print ('mean Dice per class')
        for i, c in enumerate(meanDice_per_cls):
            print ('\t class {}: {}'.format(i,c))
        print ('-'*20)
        
        data = {'mean_IU': '%.2f' % (meanIU), 
        'mean_IU_cls': ['%.2f'%(a) for a in meanIU_per_cls.tolist()],
        'mean_Dice': '%.2f' % (mean_dice), 
        'mean_Dice_cls': ['%.2f'%(b) for b in meanDice_per_cls.tolist()],
        'mean_Pixel_Accuracy': '%.2f' % (mean_pix_acc)
        }
        self.score_history['%.10d' % suffix] = data
        json.dump(self.score_history, open(os.path.join(self.path, 'meanIU{}.json'.format(self.suffix)),'w'), indent=2, sort_keys=True)


def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''


    check_size(eval_segm, gt_segm)

    # LATTER!!!
    # cl, n_cl   = union_classes(eval_segm, gt_segm)
    cl = [0,1]
    n_cl = 2 

    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = gt_segm


    sum_n_ii = 0
    sum_t_i  = 0

    for i, c in enumerate(cl):
        if i != 0:
            curr_eval_mask = eval_mask[ :, :, i]
            curr_gt_mask = gt_mask[ :, :, i]

            sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            sum_t_i  += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = float(sum_n_ii) / float(sum_t_i)
    return pixel_accuracy_

       
def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii) / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    # LATTER!!!
    # cl, n_cl   = union_classes(eval_segm, gt_segm)
    cl = [0,1]
    n_cl = 2 

    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = gt_segm

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[ :, :, i]
        curr_gt_mask = gt_mask[ :, :, i]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)
 
    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_, IU

def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
 
        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_



def dice_coef_2(eval_segm, gt_segm):

    check_size(eval_segm, gt_segm)

    # LATTER!!!
    # cl, n_cl   = union_classes(eval_segm, gt_segm)
    cl = [0,1]
    n_cl = 2 
    smooth = 1.

    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = gt_segm

    Dice = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[ :, :, i]
        curr_gt_mask = gt_mask[ :, :, i]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        y_true_f = curr_eval_mask.flatten()
        y_pred_f = curr_gt_mask.flatten()
        intersection = np.sum(y_true_f * y_pred_f)

        Dice[i] =(2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
 
    mean_dice_ = np.sum(Dice) / n_cl_gt
    return mean_dice_, Dice


def dice_coef(y_true, y_pred, smooth=1.):
    check_size(y_true, y_pred)
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return  K.mean( (2. * intersection + smooth) / (union + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)
    print cl, n_cl
    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((h, w, n_cl))

    for i, c in enumerate(cl):
        masks[ :, :, i] = segm == c

    return masks

def combine_channels(img):
    channel_num = img.shape[2]
    img_single = np.zeros((img.shape[0], img.shape[1]))
    for ch in range(channel_num):
        img_single += img[:,:,ch]
    return img_single

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise ValueError('Uneuqal image %s and mask %s size' %((h_e, w_e),(h_g, w_g)))
