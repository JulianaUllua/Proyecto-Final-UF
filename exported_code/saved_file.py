import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt


filename_0 = r'C:\Users\trini\Pictures\lena.png'
img_0 = cv2.imread(filename_0)

img_1 = cv2.cvtColor(img_0,6)

cv2.imshow('Imagen Resultado', img_1)
cv2.waitKey(0)
