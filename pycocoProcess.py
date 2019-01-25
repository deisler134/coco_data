"""
 cocodata operation
"""

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io

import IPython
import os
import json
import random
import numpy as np
import requests
from io import BytesIO
from math import trunc
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw

import multiprocessing as mp


# coco datasets operator

class CocoDataOperator():
    def __init__(self, dataDir, DataType = 'train2017', annFile):
	self.imageDir = os.path.join(dataDir, DataType) 
	self.annFile = COCO(annFile)
	self.cats = self.load_Cats()
	self.imgIds = []
    
    # load all categories
    def load_Cats():
	return self.annFile.loadCats(self.annFile.getCatIds())

    # load all images containing given categories
    def load_imageIds(catlist):
	catIds = self.annFile.getCatIds(catNms = catlist);
	imgIds = self.annFile.getImgIds(catIds = catIds );
	return imgIds

    # load local image
    def load_image(imgIds = self.imgIds):
	imgIds = self.annFile.getImgIds(imgIds = imgIds)
	imginfo = self.annFile.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
	img = io.imread("%s/%s"%(self.imageDir,imginfo['file_name']))
	return img

    def get_blend_img(img, catIds, inv = False):
	annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
	anns = coco.loadAnns(annIds)

	# load and display image
	# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
	    
	# use url to load image
	I = io.imread(img['coco_url'])

	#coco.showAnns(anns)

	mask = np.zeros( (I.shape[0], I.shape[1], 3) )

	for ann in anns:
	    m = coco.annToMask(ann)
	    img = np.ones( (m.shape[0], m.shape[1], 3) )
	    if ann['iscrowd'] == 1:
		color_mask = np.array([2.0,166.0,101.0])/255
	    if ann['iscrowd'] == 0:
		color_mask = np.random.random((1, 3)).tolist()[0]
	    for i in range(3):
		img[:,:,i] = color_mask[i]

	    m = np.dstack((0.5*m,0.5*m,0.5*m))
	    if inv == True:
		mask += np.where(m,m,img*255)
	    else:
		mask += np.where(m,img*255,m)

	    if len(I.shape) != 3:
		I = np.dstack((I,I,I))
		
	 blender = (I*0.3 + 0.7*mask).astype(np.uint8)
	 I = (np.where(mask,blender,I))

	 return I

    def multiprocessing(target, args, threadnum):
	pool = mp.Pool(processes = threadnum)
	results = [ pool.apply(target, args = args)]

	return results	

