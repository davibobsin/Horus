import cv2
from statistics import mean
import numpy as np
import glob

CALIB_PARAM_FILE = "../config/calib_param.cnf"
TABLE_PARAM_FILE = "../config/table_param.cnf"
IMAGE_FILE = '../config/temp/Base.png'
MESA_LARGURA = 445-12      # Medida em mm   
MESA_PROFUNDIDADE = 643-18 # Medida em mm
CAMERA_ID = 0

SUP = 0
INF = 1
DIR = 2
ESQ = 3

def salvar_parametros(ptos1,ptos2,H):
    file = open(TABLE_PARAM_FILE, "w+")
    file.write("P1=np.array(" + str(ptos1.tolist()) + ")\n")
    file.write("P2=np.array(" + str(ptos2.tolist()) + ")\n")
    file.write("H=np.array(" + str(H.tolist()) + ")")

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
    NK[0,0]=K[0,0]/2
    NK[1,1]=K[1,1]/2
    # Just by scaling the matrix coefficients!
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), NK, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def filtro(img1):
    # Converter imagem para HSV
    img_HSV = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)

    # Filtro amarelo
    up_ylw = np.array([40, 255, 255])
    lw_ylw = np.array([30, 100, 70])
    img_yellow = cv2.inRange(img_HSV,lw_ylw,up_ylw)*255
    
    # Filtro verde
    up_grn = np.array([80, 255, 255])
    lw_grn = np.array([70, 60, 60])
    img_green = cv2.inRange(img_HSV,lw_grn,up_grn)*255
    
    # Filtro azul
    up_blu = np.array([120, 255, 255])
    lw_blu = np.array([100, 100, 70])
    img_blue = cv2.inRange(img_HSV,lw_blu,up_blu)*255


    return img_yellow,img_green,img_blue

def bordas(img1,foto):
    # Criar imagem em branco e obter informações de tamanho da imagem
    img = np.zeros(img1.shape, np.uint8)
    lar = img.shape[1]
    alt = img.shape[0]

    # lista de pontos
    ptsX = np.array([])
    ptsY = np.array([])
    
    # Encontrar último elementos de uma linha
    # LINHA DA DIREITA
    for i in range(alt):
        fn = 1 #flag null: não existem pontos nessa linha
        for j in range(lar):
            if img1[i][j] > 0:
                fn = 0;
                last = j
        if fn!=1:
            img[i,last] = 255
            ptsX = np.append(ptsX,i)
            ptsY = np.append(ptsY,last)

    X = ptsX[0:500]
    Y = ptsY[0:500]
    m_d=(mean(X)*mean(Y)-mean(X*Y))/(mean(X)*mean(X)-mean(X*X))
    b_d = mean(Y)-m_d*mean(X)

    # lista de pontos
    ptsX = np.array([])
    ptsY = np.array([])
    
    # LINHA DA ESQUERDA
    for i in range(alt):
        fn = 1 #flag null: não existem pontos nessa linha
        for j in reversed(range(lar)):
            if img1[i][j] > 0:
                fn = 0;
                last = j
        if fn!=1:
            img[i,last] = 255
            ptsX = np.append(ptsX,i)
            ptsY = np.append(ptsY,last)

    X = ptsX[100:300]
    Y = ptsY[100:300]
    m_e=(mean(X)*mean(Y)-mean(X*Y))/(mean(X)*mean(X)-mean(X*X))
    b_e = mean(Y)-m_e*mean(X)
            
    # Encontrar último elemento de uma coluna
    # lista de pontos
    ptsX = np.array([])
    ptsY = np.array([])
    
    # LINHA DE BAIXO
    for i in range(lar):
        fn = 1 #flag null: não existem pontos nessa linha
        for j in range(alt):
            if img1[j][i] > 0:
                fn = 0;
                last = j
        if fn!=1:
            img[last,i] = 100
            ptsX = np.append(ptsX,i)
            ptsY = np.append(ptsY,last)
    
    X = ptsX[150:200]
    Y = ptsY[150:200]
    m_b=(mean(X)*mean(Y)-mean(X*Y))/(mean(X)*mean(X)-mean(X*X))
    b_b = mean(Y)-m_b*mean(X)

    pt_eb_x = int((b_e+b_b/m_b)/(m_e-1/m_b))
    pt_eb_y = int(b_e+m_e*pt_eb_x)
    pt_db_x = int((b_d+b_b/m_b)/(m_d-1/m_b))
    pt_db_y = int(b_d+m_d*pt_db_x)
    print(pt_eb_x)
    print(pt_eb_y)
    print(pt_db_x)
    print(pt_db_y)

    cv2.line(foto,(int(b_e),0),(int(b_e+m_e*1000),1000),(0,255,0),1)
    cv2.line(foto,(int(b_d),0),(int(b_d+m_d*1000),1000),(0,255,0),1)
    cv2.line(foto,(0,int(b_b)),(1000,int(b_b+m_b*1000)),(0,255,255),1)
    cv2.circle(foto,(pt_eb_y,pt_eb_x), 2, (0,0,255), -1)
    cv2.circle(foto,(pt_db_y,pt_db_x), 2, (0,0,255), -1)
    
    return img

def linhas(img1,img2):
    img = np.zeros(img1.shape)
    
    lines = cv2.HoughLines(img1,1,np.pi/180,200)
    print(lines)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)


    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(img1,1,np.pi/180,100,minLineLength,maxLineGap)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(img2,(x1,y1),(x2,y2),(0,255,0),2)
    return img2

def inverte(imagem):
    img = (255-imagem.copy())
    return img

def linhas(img0,img2):
    img = img0.copy()
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    cv2.imshow('edges',edges)
    minLineLength = 10
    maxLineGap = 20
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    print(lines)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(img2,(x1,y1),(x2,y2),(0,255,0),2)
        
    return img2
def linha_min_quadrados(img0):
    nonZero = np.squeeze(cv2.findNonZero(img0))
    
    X = nonZero[:,1].astype(float)
    Y = nonZero[:,0].astype(float)

    m =(mean(X)*mean(Y)-mean(X*Y))/(mean(X)*mean(X)-mean(X*X))
    b = mean(Y)-m*mean(X)
    return m,b
##
##    pt_eb_x = int((b_e+b_b/m_b)/(m_e-1/m_b))
##    pt_eb_y = int(b_e+m_e*pt_eb_x)
##    pt_db_x = int((b_d+b_b/m_b)/(m_d-1/m_b))
##    pt_db_y = int(b_d+m_d*pt_db_x)
##    print(pt_eb_x)
##    print(pt_eb_y)
##    print(pt_db_x)
##    print(pt_db_y)

def remove_blobs(img):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 1000

    #your answer image
    img2 = np.zeros((output.shape),dtype='uint8')
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2

def split_mid(img_hor,img_ver,mid_point_hor,mid_point_ver):
    img_up = np.zeros(img_hor.shape,dtype='uint8')
    img_dw = np.zeros(img_hor.shape,dtype='uint8')
    img_lf = np.zeros(img_ver.shape,dtype='uint8')
    img_rt = np.zeros(img_ver.shape,dtype='uint8')
    img_up[:,:mid_point_hor] = img_hor[:,:mid_point_hor]
    img_dw[:,mid_point_hor:] = img_hor[:,mid_point_hor:]
    img_lf[:mid_point_ver,:] = img_ver[:mid_point_ver,:]
    img_rt[mid_point_ver:,:] = img_ver[mid_point_ver:,:]
    cv2.imwrite('Cima.jpg',img_up)
    cv2.imwrite('Baixo.jpg',img_dw)
    cv2.imwrite('Esquerda.jpg',img_lf)
    cv2.imwrite('Direita.jpg',img_rt)
    return img_up,img_dw,img_lf,img_rt

def mid_points(img,axis=0):
    # Axis = 0 : linha na horizontal
    # Axis = 1 : Linha na vertical

    res = np.zeros(img.shape,dtype='uint8')
    m_X =0
    m_Y =0
    m_XY=0
    m_XX=0
    c=0
    for k in range(img.shape[axis]):
        media=0
        n=0
        flag=0
        for l in range(img.shape[1-axis]):
            i = k
            j = l
            if img[i][j] > 0:
                flag=1
                n=n+1
                c=c+1
                media = (media*(n-1)+j)/n
        if flag:
            m_X = (m_X*(c-1)+i)/c
            m_Y = (m_Y*(c-1)+media)/c
            m_XY = (m_XY*(c-1)+media*i)/c
            m_XX = (m_XX*(c-1)+i*i)/c
        res[i][int(media)] = 255
    print(c)
    a = (m_X*m_Y-m_XY)/(m_X*m_X-m_XX)
    b = m_Y-a*m_X
    return res,a,b

def corner_points(m,b):
    rect2 = np.zeros((4, 2), dtype = "float32")
    rect = np.zeros((4, 2), dtype = "float32")
    for i in range(2):
        for j in range(2,4):
            x_pto = (b[i]-b[j])/(m[j]-m[i])
            y_pto = b[i]+m[i]*x_pto
            y_pto1 = b[j]+m[j]*x_pto
            rect2[j-2+2*i]= (int(y_pto),int(x_pto))
            
    rect[0] = rect2[1]
    rect[1] = rect2[0]
    rect[2] = rect2[2]
    rect[3] = rect2[3]

    return rect

def mouse2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img_flat,(x,y), 2, (0,0,255), -1)
        cv2.imshow('Flat',img_flat)
        #print('X1:'+str(x)+' Y1:'+str(y))
        # Find corner points
        pts2 = np.matmul(np.linalg.inv(h),np.array([x,y,1]).T)
        x_offs = 9
        y_offs = 8
        pts_mesa = (pts2[0]/pts2[2]-73+x_offs,pts2[1]/pts2[2]-118+y_offs)
        pts_draw = (int(pts2[0]/pts2[2]),int(pts2[1]/pts2[2]))
        print(pts_mesa)
        cv2.circle(img,pts_draw, 2, (0,0,255), -1)
        cv2.imshow('image',img)

def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y), 2, (0,255,0), -1)
        cv2.imshow('image',img)
        #print('X1:'+str(x)+' Y1:'+str(y))
        # Find corner points
        pts2 = np.matmul(h,np.array([x,y,1]).T)
        x_offs = 9
        y_offs = 8
        pts_mesa = (pts2[0]/pts2[2]-73+x_offs,pts2[1]/pts2[2]-118+y_offs)
        pts_draw = (int(pts2[0]/pts2[2]),int(pts2[1]/pts2[2]))
        print(pts_mesa)
        cv2.circle(img_flat,pts_draw, 2, (0,255,0), -1)
        cv2.imshow('Flat',img_flat)

def draw_progress(img,percent):
    lar = 400
    alt = 50
    progress = np.zeros(img.shape,dtype='uint8')
    mid_x = img.shape[1]/2
    mid_y = img.shape[0]/2
    fim_x = int(mid_x-lar/2+lar*percent/100)
    cv2.rectangle(progress, (int(mid_x-lar/2),int(mid_y-alt/2)), (fim_x,int(mid_y+alt/2)), (0,255,0),-1)
    cv2.rectangle(progress, (int(mid_x-lar/2),int(mid_y-alt/2)), (int(mid_x+lar/2),int(mid_y+alt/2)), (255,255,255),3)
    img2 = cv2.addWeighted(img,0.3,progress,1,0)
    return img2
global img,h,img_flat

def bw2rgb(img):
    img2 = np.zeros((img.shape[0],img.shape[1],3),dtype='uint8')
    for i in range(3):
        img2[:,:,i] = img[:,:]
    return img2

ler_parametros()
#print('K:'+str(K)+' DIM:'+str(DIM)+' D:'+str(D))

##images = glob.glob('../img/fotos2/fotos*.jpg')
##i=0
##for fname in images:
##    img = cv2.imread(fname)
##    img = corrigir(img)
##    cv2.imwrite('../img/fotos2/foto'+str(i)+'.jpg',img)
##    i=i+1

##vid = cv2.VideoCapture(0)
##vid.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
##vid.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

    
##    break
# Tira foto

#img = cv2.imread('../img/fotos3/FotoTarugo5.jpg')

img = cv2.imread(IMAGE_FILE)

#ret,img = vid.read() 
#img = corrigir(img)

# Filtra e separa em verdem, azul e amarelo
img_amarelo,img_verde,img_azul = filtro(img)

# Remove blobs
img_amarelo = remove_blobs(img_amarelo)
img_azul = remove_blobs(img_azul)
#cv2.imshow('Cassião',cv2.resize(img_amarelo,(1000,600)))

# FALTA FAZER AINDA
# Encontra aproximadamente o meio da mesa
x_meio = 1000
y_meio = 600

# Separa parte superior, inferior, direita e esquerda
img_arr = [0,0,0,0]
img_arr[SUP],img_arr[INF],img_arr[DIR],img_arr[ESQ] = split_mid(img_amarelo,img_azul,x_meio,y_meio)

# Encontra linhas
m = [0,0,0,0]
b = [0,0,0,0]

for i in range(4):
    img_wait = draw_progress(cv2.resize(img,(1000,600)),i*25)
    cv2.imshow('image',img_wait)
    cv2.waitKey(2)
    m[i],b[i] = linha_min_quadrados(img_arr[i])
    cv2.line(img,(int(b[i]),0),(int(b[i]+m[i]*10000),10000),(0,0,255),2,3)

# Encontra os pontos da mesa
ptos = corner_points(m,b)
for pto in ptos:
    cv2.circle(img,(pto[0],pto[1]),5,(0,255,0),-1)

# Ajusta a perspectiva da imagem
maxWidth = MESA_LARGURA
maxHeight = MESA_PROFUNDIDADE

dst = np.array([
	[0, 0],
	[maxWidth - 1, 0],
	[maxWidth - 1, maxHeight - 1],
	[0, maxHeight - 1]], dtype = "float32")

M = cv2.getPerspectiveTransform(ptos, dst)
img_flat = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
##cv2.namedWindow('Flat')
##cv2.setMouseCallback('Flat',mouse2)
##cv2.imshow('Flat',img_flat)

# Encontra a Matriz Homografica
points1 = np.zeros((4, 2), dtype=np.float32)
points1[0, :] = [0,0]
points1[1, :] = [MESA_LARGURA,0]
points1[2, :] = [MESA_LARGURA,MESA_PROFUNDIDADE]
points1[3, :] = [0,MESA_PROFUNDIDADE]

h, mask = cv2.findHomography(ptos, points1, cv2.RANSAC)

# Salvar dados no arquivo 'table_param.cnf'
salvar_parametros(ptos,points1,h)

##cv2.namedWindow("image")
##cv2.setMouseCallback("image", mouse)
##cv2.imshow('image',cv2.resize(img,(1000,600)))
##
##cv2.waitKey(0)
cv2.destroyAllWindows()
