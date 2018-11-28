import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob
import io
import tkinter
import PIL.Image, PIL.ImageTk
import shutil

# Defines
calib_image_folder = '../config/imgsCalib/'
calib_param_file = "../config/calib_param.cnf"
NUM_CAMERA = 0

def chess_corners_draw(frame):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
        
    img = frame.copy()
    # Encontrar bordas xadrez na imagem
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # Se encontrou mostrar
    if ret == True:
        print('teste')
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)

    return img

def param_clean():
    open(calib_param_file, 'w').close()

def param_add(string):
    file = open(calib_param_file, "a+", encoding="utf-8")
    file.write(string)
    file.close()

def calibrar():
    CHECKERBOARD = (6,9)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(calib_image_folder+'*.jpg')
    #images = glob.glob('../img/samples/*.jpg')
    
    for fname in images:
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
        
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    param_clean()
    param_add("Found " + str(N_OK) + " valid images for calibration\n")
    param_add("DIM=" + str(_img_shape[::-1])+"\n")
    param_add("K=np.array(" + str(K.tolist()) + ")\n")
    param_add("D=np.array(" + str(D.tolist()) + ")")

class FinishWindow:
    def __init__(self,window,parent):
        self.window = window
        self.window.title("Calibração terminada")

        # Definir quem é o objeto Pai
        self.parent = parent

        # Realiza o cálculo dos parâmetros
        try:
            calibrar()
            mensagem = "Calibração completa com sucesso!"
        except:
            mensagem = "Ocorreu um erro durante a calibração, por favor, refaça a calibração!"
    
        # Texto da janela
        self.texto = tkinter.Label(window, text=mensagem)
        self.texto.pack()

        # Botão da janela
        self.btn_OK = tkinter.Button(window, text="OK PORRA!",command=self.quit)
        self.btn_OK.pack()
        
        self.window.mainloop()

    def quit(self):
        self.parent.window.destroy()
        self.window.destroy()


class App:
    def __init__(self,window,window_title,video_source=NUM_CAMERA):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # Cria objeto de captura
        self.vid = VideoCapture(self.video_source)

        # Cria área na janela para imprimir a imagem
        self.canvas = tkinter.Canvas(window,width=960,height=540)
        self.canvas.pack()

        # Cria variável de contagem do número de fotos
        self.contagem = 1

        # Cria botão para tirar fotos
        self.btn_snapshot=tkinter.Button(window, text="Gravar!", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        # Seta variável e método para atualização da imagem
        self.delay = 15
        self.update()

        # Limpa o diretório de imagens
        if os.path.exists(calib_image_folder):
            shutil.rmtree(calib_image_folder)
        os.makedirs(calib_image_folder)
        
        self.window.mainloop()

    def snapshot(self):
        ret,frame = self.vid.get_frame()
        if ret:
            file_name = calib_image_folder+"img"+str(self.contagem)+".jpg"
            cv2.imwrite(file_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.contagem = self.contagem+1
            if self.contagem == 6:
                FinishWindow(tkinter.Tk(),self)
        
    def update(self):
        ret, frame = self.vid.get_frame()
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,50)
        fontScale              = 1
        fontColor              = (255,0,0)
        lineType               = 2
        
        cv2.putText(frame, "IMAGEM "+str(self.contagem),
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
        
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.resize(frame,(960,540))))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        
        self.window.after(self.delay, self.update)

class VideoCapture:
    def __init__(self,video_source=NUM_CAMERA):
        # Abre a fonte de vídeo
        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
        
        if not self.vid.isOpened():
            raise ValueError("Não foi possível abrir a câmera", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret,frame = self.vid.read()
            if ret:
                #frame = chess_corners_draw(frame)
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
        
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

App(tkinter.Tk(),"Fotos para calibração")
