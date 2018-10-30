import cv2
import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

def filtro_branco(img1):
    img = img1.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of white color in HSV
    # change it according to your need !
    sensitivity = 60
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)

    # Remove noise dots
    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

    print(type(opening))

    #threshold to binary
    cv2.threshold(opening, 100, 255, cv2.THRESH_BINARY, opening)

    print(type(opening))

    return opening

def vertice3(img1):

    data = vertices(img1)
    cv2.imwrite('data.png', data)
    xy = peak_local_max(data, min_distance=2,threshold_abs=1500)
    # print(xy)

    black = np.zeros(data.shape, dtype = "uint8")
    # print(data.shape, black.shape)
    for pos in xy:
        # print(pos)
        black[pos[0]][pos[1]] = 255

    cv2.imwrite('result.png', black)

def vertice2(img1):
    neighborhood_size = 5
    threshold = 1500

    data = vertices(img1)

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    pos_arr = []
    for dy,dx in slices:
        x_center = int((dx.start + dx.stop - 1)/2)
        y_center = int((dy.start + dy.stop - 1)/2)
        pos_arr.append((x_center, y_center))

    black = np.zeros(data.shape, dtype = "uint8")
    print(data.shape, black.shape)
    for pos in pos_arr:
        print(pos)
        black[pos[0]][pos[1]] = 255

    cv2.imwrite('result.png', black)
    cv2.imwrite('data.png', data)

def vertices(img1):
    img = img1.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,5,5,0.1)

    cv2.imwrite('dst.png', dst)
    
    #result is dilated for marking the corners, not important
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    print(kernel)
    # dst = cv2.dilate(dst, kernel)
    # dst = cv2.erode(dst, kernel)

    # Threshold for an optimal value, it may vary depending on the image.
    # corner_y=np.where(dst>0.015*dst.max())[0]
    # corner_x=np.where(dst>0.015*dst.max())[1]
    # img[dst>0.1*dst.max()]=[0,0,255]
    # cv2.line(img,(corner_x[0],corner_y[0]),(corner_x[100],corner_y[100]),(0,255,0),2)
    return dst

# def reconhecer_linhas(img1):
#     try:
#         img = img1
#         minLineLength = 2
#         maxLineGap = 1000
#         lines = cv2.HoughLinesP(img,1,np.pi/180,200,maxLineGap,minLineLength)
#         for line in lines:
#             for x1,y1,x2,y2 in line:
#                 cv2.line(img1,(x1,y1),(x2,y2),(255,255,0),2)
#     except:
#         pass

#     return img
    

filename = '../img/caixa5.jpg'
img = cv2.imread(filename)

img = filtro_branco(img)

cv2.imwrite('non_resized.png', img)
img = cv2.resize(img, (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
cv2.imwrite('resized.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
img = vertice3(img)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

