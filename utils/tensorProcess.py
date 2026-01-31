import torch
import numpy as np 
from skimage import io as img_io
import torch.nn.functional as F

    
#from utils.auxilary_functions import image_resize, centered
import os
from utils.config import *

def torchProcess(img):
    
    #img = torch.from_numpy(img)

    #print("\n\t\t 0.min:",img.min().item(),"\t max:",img.max().item())

    #img = img.float() #/ 255.0
    
    print("\n\t\t 1.min:",img.min().item(),"\t max:",img.max().item()," len:",len(img),"\t type:",type(img))
    
    img = 1.0 - img
        
    
    #print("\n\t\t 2.min:",img.min().item(),"\t max:",img.max().item())

    
    if img.dim()==4:
        img = tensor_resize(img,height=img.shape[2] // 2,width=None)
    elif img.dim()==3:
        img = tensor_resize(img,height=img.shape[1] // 2,width=None)
    elif img.dim()==2:
        img = tensor_resize(img,height=img.shape[0] // 2,width=None)


    #img = tensor_resize(img)
    
    return img

def tensor_resize(img_tensor, height=None, width=None):
    
    if img_tensor.dim()>2:
        _, channels, old_height, old_width = img_tensor.size()
    elif img_tensor.dim()==2:
        old_height, old_width = img_tensor.size()
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.unsqueeze(0)
                
    if height is not None and width is None:
        scale = float(height) / float(old_height)
        width = int(scale * old_width)
    
    if width is not None and height is None:
        scale = float(width) / float(old_width)
        height = int(scale * old_height)
    
    img_tensor = F.interpolate(img_tensor, size=(height, width), mode='bilinear', align_corners=False)
    
    return img_tensor


def tensor_centered(word_img_tensor, target_size, centering=(.5, .5), border_value=0.0):

    #print("\n\t word_img_tensor.shape:",word_img_tensor.shape)
    if word_img_tensor.dim()==4:
        _, _, old_height, old_width = word_img_tensor.size()
    elif word_img_tensor.dim()==3:
        _, old_height, old_width = word_img_tensor.size()
    elif word_img_tensor.dim()==2:
        old_height, old_width = word_img_tensor.size()
    
    diff_h = target_size[0] - old_height
    diff_w = target_size[1] - old_width
    
    ys, ye = abs(diff_h) // 2, old_height - (abs(diff_h) - abs(diff_h) // 2)
    xs, xe = abs(diff_w) // 2, old_width - (abs(diff_w) - abs(diff_w) // 2)
    
    padh = (ys, abs(diff_h) - ys)
    padw = (xs, abs(diff_w) - xs)
    
    padded_img = F.pad(word_img_tensor, pad=(padw[0], padw[1], padh[0], padh[1]), value=border_value)
    
    return padded_img

from PIL import Image
def dumpImages(allTensors,location,nm):
    
    for tempIndx,tempImag in enumerate(allTensors):
        tempImag = tempImag.permute(1,2,0)

        tempImag = tempImag.cpu().numpy()
        tempImag = tempImag * 255
        tempImag = tempImag.astype(np.uint8)
        
        img_io.imsave(location+str(tempIndx)+"_tensor_"+nm+".png",tempImag) 