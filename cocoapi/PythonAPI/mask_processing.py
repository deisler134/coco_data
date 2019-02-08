'''
Created on Feb. 5, 2019

save image with mask

@author: deisler
'''
import glob
import cv2 as cv
import numpy as np
import os

debug_mode = True

imagepath = '../Data/project/humanparsing/JPEGImages/'
maskpath = '../Data/project/humanparsing/SegmentationClassAug/'

def get_fileslist( imagepath = imagepath ):
    print(os.path.join(imagepath, '*'))
    return glob.glob( os.path.join(imagepath, '*'))

def get_imagename( imagepath ):
    
    return imagepath.split('/')[-1]

def get_colormask( image ):
    if len(image.shape) == 3:
        mask = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        mask = image.copy()
#     thresh = cv.adaptiveThreshold(mask,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    retval, thresh = cv.threshold(mask,1, 255,cv.THRESH_BINARY)
    
    _, contours, hierarchy = cv.findContours(
       thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    mask_img = np.zeros((mask.shape[0],mask.shape[1],3), np.uint8)
    mask_img = cv.drawContours(mask_img, contours, -1, (0,220,0), -1)
    
#     if debug_mode:
#         cv.imshow('mask', mask)
#         cv.imshow('thresh', thresh)
#         cv.imshow('maskimg', mask_img)
#         cv.waitKey(0)
        
    return mask_img

def mask_on_image(imagepath, maskpath):
    
    img = cv.imread(imagepath)
    mask = cv.imread(maskpath)
    
    if (img.size != 0) and (mask.size != 0):
        mask = get_colormask( mask )
        
    maskimage = np.where(mask, img*0.3 + mask*0.7, img).astype(np.uint8)
    
#     if debug_mode:
#         cv.imshow('maskimage', maskimage)
#         cv.waitKey(0)
        
    return maskimage
    
def save_maskon_image(imagepath = imagepath, maskpath = maskpath):
    
    imagelist = get_fileslist(imagepath)
    masklist = get_fileslist(maskpath)
    
    print( imagelist[1],masklist[1])
    
    for imgpath in imagelist:
        imagename = get_imagename(imgpath)
        maskname = imagename[:-3] + 'png'
        mkpath = os.path.join(maskpath, maskname)
        
        print(imgpath,mkpath)
        
        maskimage = mask_on_image(imgpath, mkpath)
        blendimgpath = os.path.join(imagepath, imagename[:-4]+'_mask.jpg')

        cv.imwrite(blendimgpath, maskimage)
           
        
save_maskon_image(imagepath,maskpath)            
            
        
