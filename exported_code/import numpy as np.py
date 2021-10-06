import numpy as np
import os
from cv2 import cv2
from matplotlib import pyplot as plt
from pathlib import Path
filename = r'C:\Users\trini\OneDrive\Favaloro\Tesis\Codigo\Proyecto-Final-UF-1\saved_images'
filename_0 = r'C:\Users\trini\Pictures\lena.png'
img_0 = cv2.imread(filename_0)

img_1 = cv2.cvtColor(img_0,6)
"""newpath = Path(__file__).parent.absolute()
print(newpath)
print(type(img_1))

cv2.imwrite(os.path.join(newpath,'waka.png'), img_1)
cv2.waitKey(0)"""
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img_0],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

#plt.figure()
#plt.imshow((img_0))
plt.show()