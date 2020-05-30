# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:23:24 2019

@author: Mohammadreza
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

img = plt.imread('Precise.jpg', format=None)
img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

plt.imshow(img,cmap='gray')
plt.show()

res, img = cv2.threshold(img,127,1,cv2.THRESH_BINARY)

#print(img)
dimensions = img.shape
height = img.shape[0]
width = img.shape[1]

Laplace = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])


#Laplace = np.flip(Laplace,(1))
#Laplace = np.flip(Laplace,(0))
#img = np.array([[5,5,5,5,5,5],[5,5,5,5,5,5],[5,5,10,10,10,10],[5,5,10,10,10,10],
#                [5,5,5,10,10,10],[5,5,5,5,10,10]])
pad_image = np.array(np.pad(img, ((2,2),(2,2)), 'constant'))
#height=6
#width=6

new_Image = np.zeros((height,width))
for i in range(height):
    for j in range(width):
        
        Apply = pad_image[i:(i+Laplace.shape[0]),j:(j+Laplace.shape[1])]
        #print(Apply)
        multiplication = np.multiply(Apply,Laplace)
        #test=np.matrix(Apply).shape
        #print(test)
    #multiplication1 = np.multiply(Apply.shape[1],width)
        summing = np.sum(multiplication)
    #summing1 = np.array(np.sum(multiplication1))
    #Test =(np.array_equal(Apply.shape[0],Apply.shape[1]))
        
    #print(summing)

        
        new_Image[i,j] = summing 
   # print(new_Image)   

    #print(img)
    #if Test == False:
        #new_edge=np.array(np.pad(Apply,((0,0),(1,1)),'constant'))
   # result = Image.fromarray((summing * 255).astype(np.uint8))
#new_image1 = Image.fromarray(new_Image[i,j], 'L')
    
                #cv2.imshow('image',new_image)
                
                #print(new_edge)
                
            
        #print(Apply)
#blur = cv2.GaussianBlur(img,(5,5),0)

#sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
#sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=5)

#cv2.imshow('photo',blur)
#cv2.imshow('photo1',img)

#plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
#plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
#plt.imshow(img,cmap='gray')
#                plt.show()
#print(img)
#print(new_Image)
        
for row in range(height):
    for col in range(width):
        if new_Image[row][col]>0:
            new_Image[row][col]=1
        else: 
            new_Image[row][col]=0
            
#print(new_Image)
plt.imshow(new_Image,cmap='gray')
plt.figure()
#plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
