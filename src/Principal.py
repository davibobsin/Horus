import cv2
import numpy as np

def reconhecer_linhas2(img1):
    img = img1
    lines = cv2.HoughLines(img,1,2*np.pi/180,100)
    for line in lines:
        for rho,theta in lines[0]:  
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))     

    cv2.line(img,(x1,y1),(x2,y2),(255,255,255),2)
    return img

def reconhecer_linhas(img1):
    img = img1
    minLineLength = 10
    maxLineGap = 20
    lines = cv2.HoughLinesP(img,1,np.pi/180,200,maxLineGap,minLineLength)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img1,(x1,y1),(x2,y2),(255,255,0),2)
            print("X1:",x1," Y1:",y1," X2:",x2," Y2:",y2)

    return img

#Importar Imagem
imagemCaixa = cv2.imread("../img/caixa5.png")

#Reconhecimento de bordas
# cv2.Canny(input_image,output_image,threshold1,threshold2,apertureSize,L2gradient)		
edges = cv2.Canny(imagemCaixa, 35, 100, 3)
#cv2.imshow('Edges',edges)

#Processamento Blur
blur = cv2.GaussianBlur(edges, (5,5) , 0)
#cv2.imshow('Blur',blur)

# Reconhecimento de Linhas
linhas = reconhecer_linhas(blur)
cv2.imshow('Original',imagemCaixa)
cv2.imshow('Linhas',linhas)
