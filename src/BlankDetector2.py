import numpy as np
import cv2 


TABLE_PARAM_FILE = "../config/table_param.cnf"
fotos = ('../img/fotos3/FotoLimpo.jpg','../img/fotos3/FotoTarugo.jpg','../img/fotos3/FotoTarugo5.jpg','../img/fotos3/FotoTarugo6.jpg')

def ler_parametros():
    global H,P1,P2
    file = open(TABLE_PARAM_FILE, "r")
    if file.mode == 'r':
        itens = file.readlines()
        i=1
        for linha in itens:
            comando = linha.replace('\n','')
            if(linha.startswith("H=")):
                H = eval(comando.replace("H=",""))
            
            if(linha.startswith("P1=")):
                P1 = eval(comando.replace("P1=",""))
                P1 = P1.astype(int)
            
            if(linha.startswith("P2=")):
                P2 = eval(comando.replace("P2=",""))
                P2 = P2.astype(int)
    file.close()

def remove_blobs(img):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 3000

    #your answer image
    img2 = np.zeros((output.shape),dtype='uint8')
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2

def lines(img):
    img2 = np.zeros((img.shape[0],img.shape[1],3),dtype="uint8")
    gray = img.copy()#cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,200,apertureSize = 3)
    minLineLength = 30
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,80,minLineLength,maxLineGap)
    for line in lines:
        coords = line[0]
        cv2.line(img2, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)
        
    cv2.imshow('linhas',img2)

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def filtro(img1):
    img_HSV = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
    # Filtro amarelo
    up_ylw = np.array([30, 255, 255])
    lw_ylw = np.array([10, 40, 40])
    img_yellow = cv2.inRange(img_HSV,lw_ylw,up_ylw)

    return img_yellow

def bw2rgb(img):
    img2 = np.zeros((img.shape[0],img.shape[1],3),dtype='uint8')
    for i in range(3):
        img2[:,:,i] = img[:,:]
    return img2

def mask(img,pts):
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts.reshape((-1,1,2))], (255,255,255))
    img2 = cv2.bitwise_and(img,mask)
    return img2

##im = cv2.imread(fotos[0])
##a = np.linspace(0.2, 2.0, num=10)
##for i,b in enumerate(a):
##    cv2.imwrite('../img/fotos3/iter/foto'+str(i)+'.jpg',adjust_gamma(im,gamma=b))

ler_parametros()


bui = cv2.imread('building.jpg')

lines(bui)


fotos = ('../img/fotos3/FotoLimpo.jpg','../img/fotos3/FotoTarugo5.jpg')

img = mask(cv2.imread(fotos[0]),P1)
img_tarugo = mask(cv2.imread(fotos[1]),P1)


mask2 = filtro(img)
mask2 = remove_blobs(mask2)
mask2 = bw2rgb(mask2)

#img2 = cv2.bitwise_and(img,mask2)

# Linhas
img1 = filtro(img_tarugo)
img1 = remove_blobs(img1)


### Filtro de cor
##img1 = filtro(img_tarugo)
##img1 = remove_blobs(img1)
##kernel = np.ones((5,5),np.uint8)
##img1 = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)
##im2, contours, hierarchy = cv2.findContours(img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


cnt = contours[0]
epsilon = 0.05*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

img1 = bw2rgb(img1)
img1 = cv2.bitwise_and(img_tarugo,img1)
cv2.drawContours(img1, contours, -1, (255,0,0), 3)
for i in approx:
    cv2.circle(img1,(i[0][0],i[0][1]),5,(0,255,0),-1)

##pto0 = [[[0,0]]]
##for pto1 in contours:
##    print((pto1[0][0][1]-pto0[0][0][1])/(pto1[0][0][0]-pto0[0][0][0]))
##    pto0 = pto1
    
cv2.imshow('igor babaca',img1)


# MOG 2
fgbg = cv2.createBackgroundSubtractorMOG2()
fgmask = fgbg.apply(img)
fgmask = fgbg.apply(img_tarugo)
fgmask = remove_blobs(fgmask)

bordas = cv2.Canny(img_tarugo,200,100)
bordas = remove_blobs(bordas)

saida = cv2.bitwise_and(bw2rgb(bordas),bw2rgb(fgmask))


#cv2.imshow('Filtro.jpg',cv2.resize(saida,(1000,600)))


##
##fgbg = cv2.createBackgroundSubtractorMOG2()
##i=0
##
##while(i<2):
##    frame = cv2.imread(fotos[i])
##    fgmask = fgbg.apply(frame)
##    cv2.imshow('frame',fgmask)
##    i=i+1
##    k = cv2.waitKey(0)
##
##edge = cv2.Canny(cv2.imread(fotos[1]),100,200)
##cv2.imshow('Edge',edge)
##lines(edge)

cv2.waitKey(0)
cv2.destroyAllWindows()
