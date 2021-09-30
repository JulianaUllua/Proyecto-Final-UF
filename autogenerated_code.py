import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt


filename_0 = r'C:\Users\trini\Pictures\lena.png'
img_0 = cv2.imread(filename_0)
filename_1 = r'C:\Users\trini\Pictures\lena.png'
img_1 = cv2.imread(filename_1)
img_2 = cv2.bitwise_and(img_1, img_0,None)