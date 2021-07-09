'''
Morphology:
    This is a method that simple and fast but this method can only use on simple image.
    Besides, the code below only focus on black lettering on white background.

Example:
    Image 00001 is white lettering on black background.
    image 00002 is black lettering on white background.
'''

import os
import cv2
import numpy as np

Path = 'Data/example'

for filename in os.listdir(Path):
    # read image
    img = cv2.imread(Path + '/' + filename)

    # convert to gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # use Sobel edge detection to produce binary image
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3) # (src, ddepth, dx, dy, ksize)
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # erosion & dilation
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    dilation = cv2.dilate(binary, element2, iterations=1)
    erosion = cv2.erode(dilation, element1, iterations=1)
    dilation2 = cv2.dilate(erosion, element2, iterations=2)

    # contour & text region
    region = []
    contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        
        area = cv2.contourArea(cnt)
        # if the area is too small, then skip it
        if area < 1000:
            continue
        
        rect = cv2.minAreaRect(cnt)
        #print('rect is: ', rect)
        
        # get the coordinate of the region of text
        box = cv2.boxPoints(rect) # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] counterclockwise
        box = np.int0(box)

        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        if height > width * 1.3:
            continue

        region.append(box)

    # write region
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2) # (img, contour, layer, color, thickness)

    # show results
    cv2.namedWindow(filename)
    cv2.imshow(filename, img)
    cv2.waitKey()
