import cv2
import numpy as np

#218.5
#139.4

FLAG_FILE = '../config/finish.flag'
MESA_LARGURA = 445-12      # Medida em mm   
MESA_PROFUNDIDADE = 643-18 # Medida em mm
OFFSET_X = 66+(263-218)
OFFSET_Y = 108+(96-139)
TABLE_PARAM_FILE = "../config/table_param.cnf"
fotos = ('../img/fotos3/FotoLimpo.jpg','../img/fotos3/FotoTarugo.jpg','../img/fotos3/FotoTarugo5.jpg','../img/fotos3/FotoTarugo6.jpg')
IMAGE_FILE = '../config/temp/Peca.png'

PATH_ESQUELETO_M157 = '../config/macros/M157.m1s'
PATH_ESQUELETO_M158 = '../config/macros/M158.m1s'
PATH_M157 = '../../macros/Mach3Mill/M157.m1s'
PATH_M158 = '../../macros/Mach3Mill/M158.m1s'

#############################################################
VAZIO=0
SELEC=1
YES=2
NOT=3
global cont
cont=0

pts = [1,0,0,0,0,0,0,0]
lcs = [(100,250),(250,250),(200,200),(50,200),(100,150),(250,150),(200,100),(50,100)]

def flag_create():
    file = open(FLAG_FILE, "w+",encoding="utf-8")
    file.write('1')
    file.close()

def flag_destroy():
    os.remove(FLAG_FILE)

def caixa():
    img = np.zeros((300,300,3),dtype="uint8")

    #Caixa Trás
    cv2.line(img,(50,100),(50,200),(255,255,255),2)
    cv2.line(img,(200,100),(200,200),(50,50,50),2)
    cv2.line(img,(50,100),(200,100),(255,255,255),2)
    cv2.line(img,(50,200),(200,200),(50,50,50),2)

    #Diagonais
    cv2.line(img,(50,100),(100,150),(255,255,255),2)
    cv2.line(img,(200,100),(250,150),(255,255,255),2)
    cv2.line(img,(50,200),(100,250),(255,255,255),2)
    cv2.line(img,(200,200),(250,250),(50,50,50),2)
    
    #Caixa frente
    cv2.line(img,(100,150),(100,250),(255,255,255),2)
    cv2.line(img,(250,150),(250,250),(255,255,255),2)
    cv2.line(img,(100,150),(250,150),(255,255,255),2)
    cv2.line(img,(100,250),(250,250),(255,255,255),2)



    return img

def cantos():
    dots = np.zeros((300,300,3),dtype="uint8")
    
    for i in range(8):
        tipo = pts[i]
        if tipo!=VAZIO:
            if tipo==SELEC:
                cor = (255,255,0)
                linha = 2
            if tipo==YES:
                cor = (0,255,0)
                linha = -1
            if tipo==NOT:
                cor = (0,0,255)
                linha = -1
            cv2.circle(dots,lcs[i],5,cor,linha)
    return dots

#############################################################

def write_m157(x,y):
    esqueleto = open(PATH_ESQUELETO_M157,"r")
    file = open(PATH_M157,"w+")
    for i,line in enumerate(esqueleto):
        if i==0 :
            file.write("Code \"G0 X"+str(x)+" Y"+str(y)+"\" 'LINHA EDITAVEL\n")
        else:
            file.write(line)
            
    file.close()
    esqueleto.close()

def write_m158(x,y,alpha):
    esqueleto = open(PATH_ESQUELETO_M158,"r")
    file = open(PATH_M158,"w+")
    for i,line in enumerate(esqueleto):
        if i==2 :
            file.write("Code \"G68 A0.0 B0.0 R"+str(alpha)+"\" 'Rotacao do eixo de coordenadas, A e B são as coordenadas X e Y do ponto de rotação e R é o angulo de rotacao em graus\n")
        else:
            if i==1 :
                file.write("Code \"G52 X"+str(x)+" Y"+str(y)+"\" 'Coordenadas do ponto zero.. DAVI ESCREVE AQUI\n")
            else:
                file.write(line)
    file.close()
    esqueleto.close()

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

def mouse(event, x, y, flags, param):
    global pts_selec,pts_peca,img,cont
    if event == cv2.EVENT_LBUTTONDOWN:
        Y1 = x-OFFSET_Y
        X1 = y-OFFSET_X
        print('Y:'+str(Y1)+' X:'+str(X1))
        cv2.circle(img,(x,y), 2, (0,0,255), -1)
        cv2.imshow('image',img)
        pts_peca[pts_selec] = [X1,Y1]
        if pts_selec>0:
            A = (pts_peca[pts_selec][1]-pts_peca[pts_selec-1][1])
            B = (pts_peca[pts_selec-1][0]-pts_peca[pts_selec][0])
            alpha = np.arctan2(A,B)
            alpha = alpha*180/np.pi-90 #meti um -90
            
        if pts_selec==1:
            write_m158(pts_peca[0][0],pts_peca[0][1],int(alpha+0.5))

        if pts_selec==3:
            mid_x = (pts_peca[0][0]+pts_peca[1][0]+pts_peca[2][0]+pts_peca[3][0])/4
            mid_y = (pts_peca[0][1]+pts_peca[1][1]+pts_peca[2][1]+pts_peca[3][1])/4
            write_m157(mid_x,mid_y)
            flag_create()
            
        pts_selec = pts_selec+1
        # CAIXA
        pts[cont]=YES
        if cont<7:
            pts[cont+1]=SELEC
        cont=cont+1
    if event == cv2.EVENT_RBUTTONDOWN:
        pts[cont]=NOT
        if cont<7:
            pts[cont+1]=SELEC
        cont=cont+1

def juntar(img1,img2):
    lar = img1.shape[0]+img2.shape[0]
    if img1.shape[1]>img2.shape[1]:
        alt = img1.shape[1]
    if img2.shape[1]>img1.shape[1]:
        alt = img2.shape[1]
    img0 = np.zeros((lar,alt,3),dtype="uint8")
    imgo
    pass

global pts_selec,pts_peca,img

# Ler imagem
img = cv2.imread(IMAGE_FILE)
pts_selec = 0
pts_peca = np.zeros((100,2))
# Ler parâmetros da mesa
ler_parametros()
# Corrigir perspectiva
P1 = P1.astype(np.float32)
P2 = P2.astype(np.float32)
P3 = np.array([
	[0, 0],
	[MESA_LARGURA - 1, 0],
	[MESA_LARGURA - 1, MESA_PROFUNDIDADE - 1],
	[0, MESA_PROFUNDIDADE - 1]], dtype = "float32")

M = cv2.getPerspectiveTransform(P1, P3)
img = cv2.warpPerspective(img, M, (MESA_LARGURA, MESA_PROFUNDIDADE))

# Criar janela de exibição
box = caixa()
aux = cv2.add(box,cantos())
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse)
cv2.imshow('image',img)


cv2.waitKey(0)
cv2.destroyAllWindows()
