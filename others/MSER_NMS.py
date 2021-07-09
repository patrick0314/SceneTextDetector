'''
MSER(Maximally Stable Extremal Regions)

NMS(Non Maximum Suppression)

'''

import os
import cv2
import numpy as np

def nms(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == 'i':
        boxes = boxes.astype('float')

    pick = []

    # choose four coordinate
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # calculate area
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # sort
    idx = np.argsort(y2)

    # delete the repeated box
    while len(idx) > 0:
        last = len(idx) - 1
        i = idx[last]
        pick.append(i)

        # find the max and min coordinate in the remaining box
        xx1 = np.maximum(x1[i], x1[idx[:last]])
        yy1 = np.maximum(y1[i], y1[idx[:last]])
        xx2 = np.minimum(x2[i], x2[idx[:last]])
        yy2 = np.minimum(y2[i], y2[idx[:last]])

        # find the overlap area ratio
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idx[:last]]

        # if ratio > threshold, then delete
        idx = np.delete(idx, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype('int')

Path = 'Data/example'

for filename in os.listdir(Path):
    # read image
    img = cv2.imread(Path + '/' + filename)

    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rect_result = img.copy()
    unregular_result = img.copy()

    # MSER
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray) # get the unregular regions of texts
    hulls = [cv2.convexHull(r.reshape(-1, 1, 2)) for r in regions]
    cv2.polylines(unregular_result, hulls, 1, (0, 255, 0))
    cv2.imshow('unregular', unregular_result)
    cv2.waitKey()

    # turn unregular regions to regular regions
    keep = []
    for hull in hulls:
        x, y, h, w = cv2.boundingRect(hull)
        keep.append([x, y, x+w, y+h])

    # NMS
    keep = np.array(keep)
    print(keep)
    keep = nms(keep, 0.3)
    print(keep)

    for (x1, y1, x2, y2) in keep: 
        cv2.rectangle(rect_result, (x1, y1), (x2, y2), (255, 255, 0), 1)

    cv2.imshow('regular', rect_result)
    cv2.waitKey()
