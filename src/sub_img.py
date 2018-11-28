import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def resize(img, height=800):
    # Resize image to given height
    rat = height / img.shape[0]
    return cv2.resize(img, (int(rat * img.shape[1]), height))

def diff(bright, dark):
    no_light = cv2.imread(dark, 0)
    flash_light = cv2.imread(bright, 0)

    sub = cv2.absdiff(flash_light, no_light)
    #sub = cv2.subtract(flash_light, no_light)

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

def article(img_name):

    # Resize and convert to grayscale
    # img = resize(cv2.imread(img_name, 0))
    img = cv2.imread(img_name, 0)
    cv2.imwrite('../img/find_blank/article_0.png',img)

    # Bilateral filter preserv edges
    img = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imwrite('../img/find_blank/article_1.png',img)

    # from second article! (https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/)

    im_in = img.copy()

    # # Threshold.
    # # Set values equal to or above 220 to 0.
    # # Set values below 220 to 255.
    
    # th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)

    # Threshold from fisrt article.
    im_th = cv2.adaptiveThreshold(im_in, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 115, 4)

    cv2.imwrite('../img/find_blank/article_2.png',im_th)

    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground.
    im_out = (im_th | im_floodfill_inv) & im_in

    # # Display images.
    #cv2.imshow("Thresholded Image", im_th)
    #cv2.imshow("Floodfilled Image", im_floodfill)
    #cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    #cv2.imshow("Foreground", im_out)
    #cv2.waitKey(0)

    # AND AGAIN!!!!!
    th, im_th = cv2.threshold(im_floodfill, 220, 255, cv2.THRESH_BINARY_INV)

    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground.
    im_out = (im_th | im_floodfill_inv)
    
    # Display images.
    #cv2.imshow("Thresholded Image", im_th)
    #cv2.imshow("Floodfilled Image", im_floodfill)
    #cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    #cv2.imshow("Foreground", im_out)
    #cv2.waitKey(0)

    # end second article!

    # Median filter clears small details
    img = cv2.medianBlur(im_out, 11)
    cv2.imwrite('../img/find_blank/article_3.png',img)

    # # Create black and white image based on adaptive threshold
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
    # cv2.imwrite('article_2.png',img)

    # # Median filter clears small details
    # img = cv2.medianBlur(img, 11)
    # cv2.imwrite('article_3.png',img)

    # Add black border in case that page is touching an image border
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    cv2.imwrite('../img/find_blank/article_4.png',img)

    edges = cv2.Canny(img, 200, 250)
    cv2.imwrite('../img/find_blank/article.png',edges)

def range(name):
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
    article('../img/find_blank/sub.png')

if __name__ == '__main__':
    main()