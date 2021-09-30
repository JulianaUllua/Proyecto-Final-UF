import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt


filename_0 = r'C:\Users\trini\Pictures\lena.png'
img_0 = cv2.imread(filename_0)

img_1 = cv2.applyColorMap(img_0,0)

img_2 = cv2.medianBlur(img_1,13)

cv2.imshow('Imagen Resultado', img_2)
cv2.waitKey(0)
