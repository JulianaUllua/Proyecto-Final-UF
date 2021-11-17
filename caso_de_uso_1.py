import numpy as np
from cv2 import cv2

#Carga de imagen
path = r'C:\Users\trini\Pictures\Shepp_logan.png'
img = cv2.imread(path)

#Suavizado de imagen
blur = cv2.blur(img,(3,3))

#Mejora de Nitidez
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(blur, -1, kernel)

#Binarización
thresh,thresh_img = cv2.threshold(blur, 70, 120, cv2.THRESH_BINARY)

#Dilatación
kernel = np.ones((2,2), np.uint8)
dilate = cv2.dilate(thresh_img,kernel)

#Erosión
kernel = np.ones((11,11), np.uint8)
erosion = cv2.erode(dilate, kernel)

#Visualizacion de imagen 
cv2.imshow('Imagen Resultado', erosion)
cv2.waitKey(0)