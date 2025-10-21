import os
import numpy
import re
import os
import shutil
from os import path
import cv2

lines=0
labelpath='./labels/'
file='converted_ 0351'
with open(labelpath+file+'.txt', 'r') as f:
    data = f.readlines()  # 將txt中所有字符串讀入data
    for i in range(len(data)): 
        data[i]=list(map(float,filter(None,re.split('[\t \n]',data[i].strip()))))
lines=len(data)
savecroppath='./save_crop/'
oriimag=cv2.imread(file+'.png')

imah,imaw,_=oriimag.shape
maskpath='./mask/'
orimask=cv2.imread(maskpath+file+'.png')
for i in range(lines):
    classs=int(data[i][0])
    xcenter=data[i][1]
    ycenter=data[i][2]
    yolow=data[i][3]
    yoloh=data[i][4]
    xmax=0.5*(yolow*imaw+xcenter*imaw*2)
    xmin=abs(yolow*imaw-xmax)
    ymax=0.5*(yoloh*imah+ycenter*imah*2)
    ymin=abs(yoloh*imah-ymax)
    cropped_image = oriimag[int(ymin):int(ymax),int(xmin):int(xmax)]
    cropped_mask = orimask[int(ymin):int(ymax),int(xmin):int(xmax)]
    if(classs==0):
        cv2.imwrite(savecroppath+'powder_uncover/crop_img/'+str(i)+file+'.png',cropped_image)
        cv2.imwrite(savecroppath+'powder_uncover/crop_mask/'+str(i)+file+'.png',cropped_mask)
    if(classs==1):
        cv2.imwrite(savecroppath+'powder_uneven/crop_img/'+str(i)+file+'.png',cropped_image)
        cv2.imwrite(savecroppath+'powder_uneven/crop_mask/'+str(i)+file+'.png',cropped_mask)

    if(classs==2):
        cv2.imwrite(savecroppath+'scratch/crop_img/'+str(i)+file+'.png',cropped_image)
        cv2.imwrite(savecroppath+'scratch/crop_mask/'+str(i)+file+'.png',cropped_mask)

    
    print(imah)
    
    # print(len(data))
    # print(data)


