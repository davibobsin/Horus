import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def resize(img, height=800):
    # Resize image to given height
    rat = height / img.shape[0]
    return cv2.resize(img, (int(rat * img.shape[1]), height))

def diff(bright, dark):
    no_light = cv2.imread(dark)
    no_light = cv2.bilateralFilter(no_light, 9, 100, 100)
    no_light = cv2.cvtColor(no_light,cv2.COLOR_BGR2HSV)
    flash_light = cv2.imread(bright)
    flash_light = cv2.bilateralFilter(flash_light, 9, 100, 100)
    flash_light = cv2.cvtColor(flash_light,cv2.COLOR_BGR2HSV)

    (channel_h, channel_s, no_light_v) = cv2.split(no_light)
    (channel_h, channel_s, flash_light_v) = cv2.split(flash_light)

    sub = cv2.absdiff(flash_light, no_light)
    #sub = cv2.subtract(flash_light_v, no_light_v)

    cv2.imwrite('../img/find_blank/sub.png',sub)

    # plt.imshow(sub, cmap = 'gray', interpolation = 'bicubic')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()

def canny(img_name):
    img = cv2.imread(img_name, 0)
    edges = cv2.Canny(img,100,200,True)

    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

def article(img_name, diff='sub.png'):

    # Resize and convert to grayscale
    # img = resize(cv2.imread(img_name, 0))
    img = cv2.imread(img_name, 0)
    cv2.imwrite('../img/find_blank/article_0.png',img)

    # Bilateral filter preserv edges
    img = cv2.bilateralFilter(img, 9, 100, 100)
    cv2.imwrite('../img/find_blank/article_1.png',img)
    
    diff_img = cv2.imread(diff, 0)

    # from second article! (https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/)

    #TESTE
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(diff_img,(555,555),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print(ret3)
    cv2.imwrite('../img/find_blank/article_2.png',th3)

    kernel = np.ones((10,10),np.uint8)
    dilation = cv2.dilate(th3,kernel,iterations = 10)
    cv2.imwrite('../img/find_blank/article_3.png',dilation)

    res = dilation & img
    ret3,res = cv2.threshold(res,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    cv2.imwrite('../img/find_blank/article_4.png',res)
    #FIM-TESTE

    im_in = res.copy()

    # Threshold from fisrt article.
    ret3,im_th = cv2.threshold(im_in,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imwrite('../img/find_blank/article_5.png',im_th)
    closing = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('../img/find_blank/article_5_1.png',opening)

    # end second article!

    # Median filter clears small details
    img = cv2.medianBlur(opening, 11)    
    cv2.imwrite('../img/find_blank/article_6.png',img)

    comp = (img & cv2.imread(img_name, 0)) | cv2.bitwise_not(img)
    cv2.imwrite('../img/find_blank/comp.png',comp)

    # Add black border in case that page is touching an image border
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    cv2.imwrite('../img/find_blank/article_7.png',img)

    edges = cv2.Canny(img, 200, 250)
    cv2.imwrite('../img/find_blank/article.png',edges)

def range_img(name):
    lower_black = np.array([0,0,0], dtype = "uint16")
    upper_black = np.array([70,70,70], dtype = "uint16")
    black_mask = cv2.inRange(frame, lower_black, upper_black)

def stitch(names):
    pics = ()
    for name in names:
        img = cv2.imread(name)
        pics = pics + (img,)
        print("@@@")
        cv2.imshow(name, img)
    cv2.waitKey(0)
    # pics = (cv2.imread(name) for name in names)
    # print(pics)
    stitcher = cv2.createStitcher(False)
    result = stitcher.stitch(pics)
    cv2.imwrite("../img/find_blank/result.png", result[1])

def main():
    #print(cv2.__version__)
    # help(cv2.createStitcher())
    # article('wt_c_1.png')
    # stitch(['c_1.jpg', 'c_2.jpg', 'c_3.jpg'])
    diff('../img/find_blank/blank.png', '../img/find_blank/aligned.png')
    article('../img/find_blank/sub.png', '../img/find_blank/sub.png')

if __name__ == '__main__':
    main()