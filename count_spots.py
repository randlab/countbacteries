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
        kernel (np.array, optional): _description_. Defaults to np.ones((10,10),np.uint8).
        kernel_clean (np.array, optional): _description_. Defaults to np.ones((10,10),np.uint8).

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

