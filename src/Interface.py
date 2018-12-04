import cv2
import sys
import numpy as np

IMAGE_FILE = '../config/temp/Imagem.png'
CALIB_PARAM_FILE = "../config/calib_param.cnf"
CAMERA_ID = 0

TECLA_ENTER = 13

def filtro(img1):
    # Converter imagem para HSV
    img_HSV = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
    
    # Filtro amarelo
    up_ylw = np.array([40, 255, 255])
    lw_ylw = np.array([30,100,70])
    img2 = cv2.inRange(img_HSV,lw_ylw,up_ylw)
    
##    for line in img2:
##        for px in line:
##            if px>0:
##                print(px)
##    cv2.imshow('hsv',img2)
    # Filtro verde
##    up_grn = np.array([80, 255, 255])
##    lw_grn = np.array([70, 60, 60])
##    img_green = cv2.inRange(img_HSV,lw_grn,up_grn)*255
    
    # Filtro azul
    up_blu = np.array([120, 255, 255])
    lw_blu = np.array([100, 100, 70])
    img2 = img2+cv2.inRange(img_HSV,lw_blu,up_blu)

    return img2

def remove_blobs(img):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 200

    #your answer image
    img2 = np.zeros((output.shape),dtype='uint8')
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2

def contorno(filt,img0):
    cnt = img0.copy()
    im2, contours_sup, hierarchy = cv2.findContours(filt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cnt, contours_sup, -1, (0,255,0), 3)
    return cnt

def ler_parametros():
    global K,DIM,D
    file = open(CALIB_PARAM_FILE, "r")
    if file.mode == 'r':
        itens = file.readlines()
        i=1
        for linha in itens:
            comando = linha.replace('\n','')
            if(linha.startswith("DIM=")):
                DIM = eval(comando.replace("DIM=",""))
            
            if(linha.startswith("K=")):
                K = eval(comando.replace("K=",""))

            if(linha.startswith("D=")):
                D = eval(comando.replace("D=",""))
    file.close()
    return itens

def corrigir(img1):
    img = img1.copy()
    h,w = img.shape[:2]
    
    NK = K.copy()
    NK[0,0]=K[0,0]*0.7
    NK[1,1]=K[1,1]*0.7
    # Just by scaling the matrix coefficients!
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), NK, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def escrever_texto(img,texto):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (300,400)
    fontScale              = 1.2
    fontColor              = (255,255,255)
    lineType               = 2
    
    cv2.putText(img, texto,
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)
    return img


if len(sys.argv)<2:
    sys.exit()
else:
    print(sys.argv[1])
IMAGE_FILE = sys.argv[1]

ler_parametros()

vid = cv2.VideoCapture(CAMERA_ID)
vid.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

flag=1
camera = CAMERA_ID
while flag:
    try:
        global img
        ret,img = vid.read()
        img = corrigir(img)
        #img = cv2.imread('../config/temp/Base.png')

        # Processa a imagem
        cont = filtro(img)
        cont = remove_blobs(cont)
        cont = contorno(cont,img)

        #Preparar imagem para mostrar
##    rows,cols,c = cont.shape
##    M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
##    dst = cv2.warpAffine(cont,M,(cols,rows))
        dst = cv2.resize(cont,(1000,600))
        
        cv2.imshow('Camera',dst)
        
    except:
        img = cv2.imread('../img/erro_camera.png')
        img = escrever_texto(img,'ERRO NA CAMERA '+str(camera))
        cv2.imshow('Camera',img)
    
    key_pressed = cv2.waitKey(100)
    
    if key_pressed>47 and key_pressed<58:
        camera = key_pressed-48
        vid = cv2.VideoCapture(camera)
        vid.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

    if key_pressed==27:
        flag=0
    
    if key_pressed==TECLA_ENTER:
        flag=0
        cv2.imwrite(IMAGE_FILE,img)
        
cv2.destroyAllWindows()
