import os
import numpy as np
import cv2


EXPAND_RANGE = {
    'WIDTH': 20,
    'HEIGHT': 20,
}


def check_embedded(bin_img, bbox):
    """
        - bbox have format: [x, y, w, h]
    """
    # Get the expand bounding box
    bbox = [bbox[0]-5, bbox[1]-5, bbox[2]+10, bbox[3]+10]
    
    expanded_bbox = [bbox[0] - int(EXPAND_RANGE['WIDTH']/2),
                     bbox[1] - int(EXPAND_RANGE['HEIGHT']/2),
                     bbox[2] + EXPAND_RANGE['WIDTH'],
                     bbox[3] + EXPAND_RANGE['HEIGHT']
    ]
    height_img, width_img = bin_img.shape

    # Loop through all pixel in expanded bbox
    for i in range(expanded_bbox[0], expanded_bbox[0]+expanded_bbox[2]):
        for j in range(expanded_bbox[1], expanded_bbox[1]+expanded_bbox[3]):
            if ((bbox[0] < i < bbox[0]+bbox[2]) and (bbox[1] < j < bbox[1]+bbox[3])) or i < 0 or j < 0 or i > width_img or j > height_img:
                continue
            if bin_img[j][i] == 0:
                return True
    return False