import cv2
import numpy as np

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

    return opening

def vertice2(img1):
    data = img1.copy()
    neighborhood_size = 5
    threshold = 1500

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)

    data[(x,y)]=[0,0,255]
    cv2.imshow('vertice',data)

def vertices(img1):
    img = img1.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,4,3,0.06)
    
    #result is dilated for marking the corners, not important
    # = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    corner_y=np.where(dst>0.015*dst.max())[0]
    corner_x=np.where(dst>0.015*dst.max())[1]
    img[dst>0.1*dst.max()]=[0,0,255]
    cv2.line(img,(corner_x[0],corner_y[0]),(corner_x[100],corner_y[100]),(0,255,0),2)
    return img

def reconhecer_linhas(img1):
    try:
        img = img1
        minLineLength = 2
        maxLineGap = 1000
        lines = cv2.HoughLinesP(img,1,np.pi/180,200,maxLineGap,minLineLength)
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img1,(x1,y1),(x2,y2),(255,255,0),2)
    except:
        pass

    return img
    

filename = '../img/caixa5.jpg'
img = cv2.imread(filename)

img = filtro_branco(img)
img = vertice2(img)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

