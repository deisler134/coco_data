'''
Created on Feb. 7, 2019

    evaluation of cocoformat_dataset_json 

@author: deisler
'''


from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


ROOT_DIR = '/notebooks/maskrcnn-benchmark/datasets/coco/human_parsing'
annFile='{}/Human_parsing_train2018.json'.format(ROOT_DIR)
print(annFile)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display human parsing categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('Human_parsing categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('Human_parsing supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds )
imgIds = coco.getImgIds(imgIds = ['2500_12']) 
img = coco.loadImgs(imgIds)[0] 
print(img,catIds)

I = io.imread('/notebooks/maskrcnn-benchmark/datasets/coco/human_parsing/JPEGImages/%s'%(img['file_name']))
plt.axis('off')
plt.imshow(I)
plt.show()

# load and display instance annotations
plt.imshow(I); plt.axis('off')

# imgIds must be list
imglist = []
imglist.append(img['id'])
annIds = coco.getAnnIds(imgIds=imglist, catIds=catIds, iscrowd=None)
print(img['id'],catIds,annIds)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)