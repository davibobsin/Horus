import numpy as np
import cv2

fotos = ('../img/fotos3/FotoLimpo.jpg','../img/fotos3/FotoTarugo.jpg','../img/fotos3/FotoTarugo5.jpg','../img/fotos3/FotoTarugo6.jpg')

def remove_blobs(img):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 10

    #your answer image
    img2 = np.zeros((output.shape),dtype='uint8')
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2

imlimpo = cv2.imread(fotos[0])
imblank = cv2.imread(fotos[1])

edgelimpo = cv2.Canny(imlimpo,100,200)
edgeblank = cv2.Canny(imblank,100,200)

im3 = edgeblank-edgelimpo
im3 = remove_blobs(im3)

cv2.imshow('Original',cv2.resize(edgelimpo,(1000,600)))
cv2.imshow('Tarugo  ',cv2.resize(edgeblank,(1000,600)))

im3 = edgeblank-edgelimpo

cv2.waitKey(0)
cv2.destroyAllWindows()


