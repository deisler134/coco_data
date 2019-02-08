'''
Created on Feb. 4, 2019

create own dataset as coco datasets format by 'pycococreatortools'

@author: deisler
'''
#!/usr/bin/env python3

import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
from PIL import Image # (pip install Pillow)



import datetime
import json
import os
import re
import fnmatch
from pycococreatortools import pycococreatortools

ROOT_DIR = '/notebooks/maskrcnn-benchmark/datasets/coco/human_parsing'
IMAGE_DIR = os.path.join(ROOT_DIR, "JPEGImages")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "SegmentationClassAug")

INFO = {
    "description": "Human parsing Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2019,
    "contributor": "Deisler",
    "date_created": datetime.datetime.now().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'person',
        'supercategory': 'person',
    },
    {
        'id': 2,
        'name': 'bicycle',
        'supercategory': 'vehicle',
    },
]

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    # from start to end match filename
    file_name_prefix = '^' + basename_no_extension + '$'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
#     files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

debug_mode = True
debugloop = 0

# categories_name correspond to files' content
categories_name = ['person']

image_id = 1
segmentation_id = 1

# filter for jpeg images
for root, _, files in os.walk(IMAGE_DIR):
    image_files = filter_for_jpeg(root, files)
    


    # go through each image
    for image_filename in image_files:
        image = Image.open(image_filename)
        
        # get image_id from filename
        image_id = image_filename.split('/')[-1]
        image_id = image_id.split('.')[0]
        
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_filename), image.size)
        coco_output["images"].append(image_info)
        
        print(image_filename)
        
        # filter for associated png annotations
        for root, _, files in os.walk(ANNOTATION_DIR):
            annotation_files = filter_for_annotations(root, files, image_filename)
            
            print(annotation_files)
            
            # go through each associated annotation
            for annotation_filename in annotation_files:

#                 print(annotation_filename)
                class_id = [x['id'] for x in CATEGORIES if x['name'] in categories_name][0]
                category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
#                 binary_mask = np.asarray(Image.open(annotation_filename)
#                     .convert('1')).astype(np.uint8)
                # why original version convert binary_mask to '1'bit? 
                binary_mask = np.asarray(Image.open(annotation_filename)).astype(np.uint8)
#                 print(binary_mask.shape)
#                 Image.open(annotation_filename).convert('1').save('/notebooks/maskrcnn-benchmark/datasets/coco/human_parsing/test.png')
                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask,
                    image.size, tolerance=2)
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)

                segmentation_id = segmentation_id + 1
                
#                 print("annotation_info",annotation_info)
#                 if debug_mode:
#                     if debugloop > 2:  
#                         break
#                     debugloop += 1
#                     print("debug:",debugloop)
#                     print(image_filename)
#                     print(image_id)
#                     print(image_info)
#         if debug_mode:
#             if debugloop > 1:
#                 break

with open('{}/Human_parsing_train2018.json'.format(ROOT_DIR), 'w') as output_json_file:
    json.dump(coco_output, output_json_file)
    
