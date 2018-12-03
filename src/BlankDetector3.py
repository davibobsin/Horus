import cv2
import numpy as np

MESA_LARGURA = 445-18      # Medida em mm   
MESA_PROFUNDIDADE = 643-18 # Medida em mm  
TABLE_PARAM_FILE = "../config/table_param.cnf"
fotos = ('../img/fotos3/FotoLimpo.jpg','../img/fotos3/FotoTarugo.jpg','../img/fotos3/FotoTarugo5.jpg','../img/fotos3/FotoTarugo6.jpg')
IMAGE_FILE = '../config/temp/Peca.png'

PATH_ESQUELETO_M157 = '../config/macros/M157.m1s'
PATH_ESQUELETO_M158 = '../config/macros/M158.m1s'
PATH_M157 = '../config/M157.m1s'
PATH_M158 = '../config/M158.m1s'

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
    global pts_selec,pts_peca,img
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y), 2, (0,0,255), -1)
        cv2.imshow('image',img)
        pts_peca[pts_selec] = [x,y]
        if pts_selec>0:
            alpha = np.arctan2((pts_peca[pts_selec-1][1]-pts_peca[pts_selec][1]),(pts_peca[pts_selec][0]-pts_peca[pts_selec-1][0]))
            alpha = alpha*180/np.pi
        if pts_selec==1:
            write_m158(pts_peca[0][0],pts_peca[0][1],int(alpha+0.5))
        
        if pts_selec==3:
            mid_x = (pts_peca[0][0]+pts_peca[1][0]+pts_peca[2][0]+pts_peca[3][0])/4
            mid_y = (pts_peca[0][1]+pts_peca[1][1]+pts_peca[2][1]+pts_peca[3][1])/4
            write_m157(mid_x,mid_y)
            
        pts_selec = pts_selec+1
              

global pts_selec,pts_peca,img
# Ler imagem
img = cv2.imread('../img/fotos3/FotoTarugo5.jpg')
pts_selec = 0
pts_peca = np.zeros((100,2))
# Ler parâmetros da mesa
ler_parametros()

# Corrigir perspectiva
P1 = P1.astype(np.float32)
P2 = P2.astype(np.float32)
M = cv2.getPerspectiveTransform(P1, P2)
img = cv2.warpPerspective(img, M, (MESA_LARGURA, MESA_PROFUNDIDADE))

# Criar janela de exibição
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse)
cv2.imshow('image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
