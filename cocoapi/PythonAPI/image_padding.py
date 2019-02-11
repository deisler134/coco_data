'''
Created on Feb. 11, 2019

    make border for images 

@author: deisler
'''

import sys
from random import randint
import cv2 as cv
from _ast import If
def main(path):
    
    borderType = cv.BORDER_CONSTANT
    window_name = "copyMakeBorder Demo"
    
    imageName = path if len(path) > 0 else 'lena.jpg'
    # Loads an image
    src = cv.imread(cv.samples.findFile(imageName), cv.IMREAD_COLOR)
    
#     if src.shape[-1] < 4:
#         cv.cvtColor(src,cv.COLOR_BGR2BGRA)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: copy_make_border.py [image_name -- default lena.jpg] \n')
        return -1
    
    print ('\n'
           '\t   copyMakeBorder Demo: \n'
           '     -------------------- \n'
           ' ** Press \'c\' to set the border to a random constant value \n'
           ' ** Press \'r\' to set the border to be replicated \n'
           ' ** Press \'ESC\' to exit the program ')
    
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    
    top = int(0.05 * src.shape[0])  # shape[0] = rows
    bottom = top
    left = int(0.05 * src.shape[1])  # shape[1] = cols
    right = left
    
    while 1:
        
        value = [randint(0, 255), randint(0, 255), randint(0, 255)]
#         value = [0, 0, 0,0]
        
        dst = cv.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)
        
        cv.imshow(window_name, dst)
        
        c = cv.waitKey(500)
        if c == 27:
            break
        elif c == 99: # 99 = ord('c')
            borderType = cv.BORDER_CONSTANT
        elif c == 114: # 114 = ord('r')
            borderType = cv.BORDER_REPLICATE
#         elif c == ord('t'):
#             borderType = cv.BORDER_TRANSPARENT 
        
    return 0
if __name__ == "__main__":

#     main(sys.argv[1:])
    main("/media/deisler/Data/project/coco/cocodata/Deep-Image-Matting/images/0_image.png")