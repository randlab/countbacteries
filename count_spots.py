import cv2
import skimage.measure as measure
import numpy as np
from typing import List

def count_spots(    bgr_img:np.array, 
                    h_range:List[float] = np.array((0.5,0.7)),
                    s_range:List[float] = np.array((0.4,1)),
                    v_range:List[float] = np.array((0,1)),
                    kernel: np.array = np.ones((10,10),np.uint8),
                    kernel_clean: np.array = np.ones((20,20),np.uint8),
                ):
    """Counts the number of bacterial spots in a picture

    Args:
        bgr_img (np.array): image buffer (bgr format)
        h_range (List[float], optional): hue range. Defaults to np.array((0.5,0.7)).
        s_range (List[float], optional): saturation range. Defaults to np.array((0.4,1)).
        v_range (List[float], optional): value range. Defaults to np.array((0,1)).
        kernel (np.array, optional): Kernel for erosion. Defaults to np.ones((10,10),np.uint8).
        kernel_clean (np.array, optional): Kernel for morphological cleaning. Defaults to np.ones((10,10),np.uint8).
    Returns:
        _type_: _description_
    """
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV )
    hmask = ((hsv_img[:,:,0]/180 > h_range[0]) & (hsv_img[:,:,0]/180<h_range[1]))
    smask = ((hsv_img[:,:,1]/255 > s_range[0]) & (hsv_img[:,:,1]/255<s_range[1]))
    vmask = ((hsv_img[:,:,2]/255 > v_range[0]) & (hsv_img[:,:,2]/255<v_range[1]))
    mask = smask&hmask&vmask
    
    erosion = cv2.erode(mask.astype(np.uint8),kernel,iterations = 1)
    erosion = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel_clean)
    erosion = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel_clean)

    labels, count = measure.label(erosion, connectivity=2, return_num=True)
    return count, labels

def find_substrate( bgr_img:np.array,
                    gkernel: np.array = (11,11),
                    minRadius:int=700,
                    maxRadius:int=1400,
                    ):
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV )
    vmat = hsv_img[:,:,2]
    blurred = np.array(cv2.GaussianBlur(vmat, (11, 11), 0)*255).astype(np.uint8)
    ret2,th2 = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    circles = cv2.HoughCircles(th2, cv2.HOUGH_GRADIENT, 1.2, 100,minRadius=700,maxRadius=1400)
    return np.mean(np.array(circles[0]),axis=0)


def get_mask(x:float,y:float,r:float,ref:np.array):
    mask = np.zeros_like(ref)
    cv2.circle(mask, (int(x), int(y)), int(r), (1,1,1), -1)
    return np.array(mask).astype(np.float32)
