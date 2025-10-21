from turtle import screensize, title
from PyQt5 import QtWidgets,QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys, UI
from detect import run
import os
import platform
import sys
import shutil
import torch
import os
import numpy
import re
from os import path
import cv2
import math
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from U_model.unet_model import UNet
import json
import time

def bb_intersection_over_union(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
TAR_HEIGHT=400
TAR_WIDTH=300
savefinalimgwithmask='./save_final/save_image_segment/'
savefinalmask='./save_final/save_segment/'
savefinalimgwithbox='./save_final/save_image_box/'
saveoimgwithbox='./save_final/save_oimage_box/'
oriimagepath='./images/'
orimaskpath='./mask/'
wei='./best.pt'
Uwei='U_model.pth'
outpath = './runs/detect/exp/'
lebelpath='./runs/detect/exp/labels/'
json_folder_path = './Json/'
save_all_fillname=[]
GTtypt=[]
PDtypt=[]
Iou=[]
Dice=[]
all_filenames = os.listdir(oriimagepath)
imgclas=['powder_uncover','powder_uneven','scratch']
imgclasadd=['UC','UE','SC']
objcls=['image','label','mask']
total_imgc=0
cur_display=0
stage=0  
detflag=0
segflag=0
class myMainWindow(QMainWindow, UI.Ui_MainWindow):
    def __init__(self):
         super().__init__()
         self.setupUi(self)
         self.pushButton.clicked.connect(self.Load_Image)
         self.pushButton_2.clicked.connect(self.Det)
         self.pushButton_3.clicked.connect(self.Seg)
         self.pushButton_4.clicked.connect(self.prev)
         self.pushButton_5.clicked.connect(self.next)
    def next(self):
        if(stage==0):
            return
        if(stage==1):
            global cur_display
            cur_display=cur_display+1
            if(cur_display>=total_imgc):
                cur_display=total_imgc
                return
            if(detflag==1):
                scene = QtWidgets.QGraphicsScene()    
                scene.setSceneRect(0, 0, 399, 349)
                imgs = QtGui.QPixmap(saveoimgwithbox+save_all_fillname[cur_display]+'.png')        
                imgs = imgs.scaled(399,349)  
                scene.addPixmap(imgs)                 
                self.graphicsView_2.setScene(scene)
                scene = QtWidgets.QGraphicsScene()    
                scene.setSceneRect(0, 0, 399, 349)
                imgs = QtGui.QPixmap(savefinalimgwithbox+save_all_fillname[cur_display]+'.png')        
                imgs = imgs.scaled(399,349)  
                scene.addPixmap(imgs)                 
                self.graphicsView_6.setScene(scene)
                self.label_8.setText("Type(GT): "+GTtypt[cur_display])
                self.label_9.setText("Predict: "+PDtypt[cur_display])
                self.label_10.setText("IoU :"+str(round(Iou[cur_display],3)))
            if(segflag==1):
                scene = QtWidgets.QGraphicsScene()    
                scene.setSceneRect(0, 0, 399,349)
                imgs = QtGui.QPixmap(savefinalmask+save_all_fillname[cur_display]+'.png')        
                imgs = imgs.scaled(399,349)
                scene.addPixmap(imgs)
                self.graphicsView_3.setScene(scene)
                scene = QtWidgets.QGraphicsScene()    
                scene.setSceneRect(0, 0, 449,399)
                imgs = QtGui.QPixmap(savefinalimgwithmask+save_all_fillname[cur_display]+'.png')        
                imgs = imgs.scaled(449,399)
                scene.addPixmap(imgs)
                self.graphicsView_4.setScene(scene)
                self.label_11.setText("Dice Coefficient: "+str(round(Dice[cur_display],3)))
                self.label_19.setText('')
            scene = QtWidgets.QGraphicsScene()    
            scene.setSceneRect(0, 0, 399,349)
            imgs = QtGui.QPixmap(oriimagepath+save_all_fillname[cur_display]+'.png')        
            imgs = imgs.scaled(399,349)  
            scene.addPixmap(imgs)                 
            self.graphicsView.setScene(scene)
            self.label_5.setText("Current Image : "+str(cur_display)+'/'+str(total_imgc))
    def prev(self):
        if(stage==0):
            return
        if(stage==1):
            global cur_display
            cur_display=cur_display-1
            if(cur_display<=0):
                cur_display=0
                return
            if(detflag==1):
                scene = QtWidgets.QGraphicsScene()    
                scene.setSceneRect(0, 0, 399, 349)
                imgs = QtGui.QPixmap(saveoimgwithbox+save_all_fillname[cur_display]+'.png')        
                imgs = imgs.scaled(399,349)  
                scene.addPixmap(imgs)                 
                self.graphicsView_2.setScene(scene)
                scene = QtWidgets.QGraphicsScene()    
                scene.setSceneRect(0, 0, 399, 349)
                imgs = QtGui.QPixmap(savefinalimgwithbox+save_all_fillname[cur_display]+'.png')        
                imgs = imgs.scaled(399,349)  
                scene.addPixmap(imgs)                 
                self.graphicsView_6.setScene(scene)
                self.label_8.setText("Type(GT): "+GTtypt[cur_display])
                self.label_9.setText("Predict: "+PDtypt[cur_display])
                self.label_10.setText("IoU :"+str(round(Iou[cur_display],3)))
            if(segflag==1):
                scene = QtWidgets.QGraphicsScene()    
                scene.setSceneRect(0, 0, 399,349)
                imgs = QtGui.QPixmap(savefinalmask+save_all_fillname[cur_display]+'.png')        
                imgs = imgs.scaled(399,349)
                scene.addPixmap(imgs)
                self.graphicsView_3.setScene(scene)
                scene = QtWidgets.QGraphicsScene()    
                scene.setSceneRect(0, 0, 449,399)
                imgs = QtGui.QPixmap(savefinalimgwithmask+save_all_fillname[cur_display]+'.png')        
                imgs = imgs.scaled(449,399)
                scene.addPixmap(imgs)
                self.graphicsView_4.setScene(scene)
                self.label_11.setText("Dice Coefficient: "+str(round(Dice[cur_display],3)))
                self.label_19.setText('')
            scene = QtWidgets.QGraphicsScene()    
            scene.setSceneRect(0, 0, 399,349)
            imgs = QtGui.QPixmap(oriimagepath+save_all_fillname[cur_display]+'.png')        
            imgs = imgs.scaled(399,349)  
            scene.addPixmap(imgs)                 
            self.graphicsView.setScene(scene)
            self.label_5.setText("Current Image : "+str(cur_display)+'/'+str(total_imgc))
        
    def Load_Image(self):
            folder=QFileDialog().getExistingDirectory(self,"Load Image")
            if(folder==''): return
            k=0
            #s 115 120 125   iou 0.945 dice 0.954
            #ue
            global total_imgc
            total_imgc=len(os.listdir(folder+'/'+imgclas[0]+'/'+objcls[0]+'/'))+len(os.listdir(folder+'/'+imgclas[1]+'/'+objcls[0]+'/'))+len(os.listdir(folder+'/'+imgclas[2]+'/'+objcls[0]+'/'))
            for i in range(3):
                for j in range(3):
                    get_files = os.listdir(folder+'/'+imgclas[i]+'/'+objcls[j]+'/')
                    if(j==0):
                        for g in get_files:
                            glen=len(g)
                            file=g[0:glen-4]
                            save_all_fillname.append((imgclasadd[i]+file))
                            self.label_19.setText('move image to dirction  and resize image '+ g +' progress: '+str(k+1)+'/'+str(total_imgc))
                            app.processEvents()
                            k=k+1
                            if(i==2):
                                shutil.copyfile(folder+'/'+imgclas[i]+'/'+objcls[j]+'/' + g, oriimagepath+imgclasadd[i]+g)
                            else:
                                ori=cv2.imread(folder+'/'+imgclas[i]+'/'+objcls[j]+'/' + g)
                                imah,imaw,_ = ori.shape
                                cv2.imwrite(oriimagepath+imgclasadd[i]+g,cv2.resize(ori,(int(imaw/3),int(imah/3)),interpolation = cv2.INTER_CUBIC))
                    elif(j==1):
                        for g in get_files:
                            shutil.copyfile(folder+'/'+imgclas[i]+'/'+objcls[j]+'/' + g, json_folder_path+imgclasadd[i]+g)
                    else:
                        for g in get_files:
                            if(i==2):
                                shutil.copyfile(folder+'/'+imgclas[i]+'/'+objcls[j]+'/' + g,orimaskpath+imgclasadd[i]+g)
                            else:
                                ori=cv2.imread(folder+'/'+imgclas[i]+'/'+objcls[j]+'/' + g)
                                imah,imaw,_ = ori.shape
                                cv2.imwrite(orimaskpath+imgclasadd[i]+g,cv2.resize(ori,(int(imaw/3),int(imah/3)),interpolation = cv2.INTER_CUBIC))
            self.label_19.setText('move image to dirction  and resize image: Done!')
            scene = QtWidgets.QGraphicsScene()    
            scene.setSceneRect(0, 0, 399,349)
            imgs = QtGui.QPixmap(oriimagepath+save_all_fillname[cur_display]+'.png')        
            self.label_5.setText("Current Image : "+str(cur_display+1)+'/'+str(total_imgc))
            imgs = imgs.scaled(399,349)  
            scene.addPixmap(imgs)                 
            self.graphicsView.setScene(scene)
            global stage
            stage=1
    def Det(self):
        Fp=[0,0,0]
        Tp=[0,0,0]
        start = time.process_time()
        if os.path.isdir(outpath):
            shutil.rmtree(outpath)
        else:
            print("目錄不存在。")
        self.label_19.setText('yolo detecting...')
        app.processEvents()
        run(source=oriimagepath,weights=wei,iou_thres=0.1) #yolov5 to 
        k1=0
        for file in save_all_fillname:
            totaliou=0
            k1=k1+1
            pros='Draw box and save img with box: '+str(k1)+'/'+str(len(save_all_fillname))
            self.label_19.setText(pros)
            app.processEvents()
            lines=0
            with open(lebelpath+file+'.txt', 'r') as f:
                data = f.readlines()  
                for i in range(len(data)): 
                    data[i]=list(map(float,filter(None,re.split('[\t \n]',data[i].strip()))))
            lines=len(data)
            oriimag=cv2.imread(oriimagepath+file+'.png')
            orimag_box=oriimag.copy()
            orimag_boxo=oriimag.copy()
            imah,imaw,_ = oriimag.shape
            with open(json_folder_path + file+'.json','r') as f:
                w = json.load(f)
            num = len(w['shapes'])
            ax=[]
            for i in range(num):
                x_min = w['shapes'][i]['points'][0][0]
                y_min = w['shapes'][i]['points'][0][1]
                x_max = w['shapes'][i]['points'][1][0]
                y_max = w['shapes'][i]['points'][1][1]
                typef = w['shapes'][i]['label']
                if(typef!='scratch'):
                    x_min=int(x_min/3)
                    x_max=int(x_max/3)
                    y_min=int(y_min/3)
                    y_max=int(y_max/3)
                if(x_min>x_max):
                    temp=x_min
                    x_min=x_max
                    x_max=temp
                if(y_min>y_max):
                    temp=y_min
                    y_min=y_max
                    y_max=temp
                    
                if(typef=='powder_uncover'):
                    orimag_boxo=cv2.rectangle(orimag_boxo,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,0,255),3)
                if(typef=='powder_uneven'):
                    orimag_boxo=cv2.rectangle(orimag_boxo,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),3)
                if(typef=='scratch'):
                    orimag_boxo=cv2.rectangle(orimag_boxo,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(255,0,0),3)
                ax.append([typef,x_min,y_min,x_max,y_max])
            ####
            for i in range(lines):
                classs=int(data[i][0])
                xcenter=data[i][1]
                ycenter=data[i][2]
                yolow=data[i][3]
                yoloh=data[i][4]
                xmax=int(0.5*(yolow*imaw+xcenter*imaw*2))
                xmin=int(abs(yolow*imaw-xmax))
                ymax=int(0.5*(yoloh*imah+ycenter*imah*2))
                ymin=int(abs(yoloh*imah-ymax))
                bestiou=0
                for item in ax:
                    targetclass=0
                    if(item[0]=='scratch'):
                        targetclass=2
                    elif(item[0]=='powder_uneven'):
                        targetclass=1
                    else:
                        targetclass=0
                    k=bb_intersection_over_union([xmin,ymin,xmax,ymax],item[1:5])
                    if((bestiou<k)&(targetclass==classs)):
                        bestiou=k
                if(targetclass==classs):
                    if(bestiou<0.5):
                        Fp[classs]+=1
                    else:
                        Tp[classs]+=1
                totaliou=totaliou+bestiou
                if(classs==0):
                    orimag_box=cv2.rectangle(orimag_box,(xmin,ymin),(xmax,ymax),(0,0,255),3)
                if(classs==1):
                    orimag_box=cv2.rectangle(orimag_box,(xmin,ymin),(xmax,ymax),(0,255,0),3)
                if(classs==2):
                    orimag_box=cv2.rectangle(orimag_box,(xmin,ymin),(xmax,ymax),(255,0,0),3)
                    continue
            averageiou=totaliou/lines
            averageiou=round(averageiou,3)
            Iou.append(averageiou)
            cv2.imwrite(savefinalimgwithbox+file+'.png',orimag_box)
            cv2.imwrite(saveoimgwithbox+file+'.png',orimag_boxo)
            
            GTtypt.append(typef)
            if(classs==0):
                PDtypt.append('uncover')
            elif(classs==1):
                PDtypt.append('uneven')
            else:
                PDtypt.append('scratch')
        end = time.process_time()
        self.label_13.setText("AP50(uncover): "+str(round(float(Tp[0])/float(Fp[0]+Tp[0]),3)))
        self.label_14.setText("AP50(uneven): "+str(round(float(Tp[1])/float(Fp[1]+Tp[1]),3)))
        self.label_15.setText("AP50(scratch): "+str(round(float(Tp[2])/float(Fp[2]+Tp[2]),3)))
        self.label_7.setText("Detection FPS: "+str(round(((end-start)/total_imgc),3)))
        self.label_19.setText('Detection Done!')
        self.label_5.setText("Current Image : "+str(cur_display+1)+'/'+str(total_imgc))
        scene = QtWidgets.QGraphicsScene()    
        scene.setSceneRect(0, 0, 399,349)
        imgs = QtGui.QPixmap(saveoimgwithbox+save_all_fillname[cur_display]+'.png')        
        imgs = imgs.scaled(399,349)  
        scene.addPixmap(imgs)                 
        self.graphicsView_2.setScene(scene)
        scene = QtWidgets.QGraphicsScene()    
        scene.setSceneRect(0, 0, 399,349)
        imgs = QtGui.QPixmap(savefinalimgwithbox+save_all_fillname[cur_display]+'.png')        
        imgs = imgs.scaled(399,349)  
        scene.addPixmap(imgs)                 
        self.graphicsView_6.setScene(scene)
        global detflag
        detflag=1
        self.label_8.setText("Type(GT): "+GTtypt[cur_display])
        self.label_9.setText("Predict: "+PDtypt[cur_display])
        self.label_10.setText("IoU :"+str(Iou[cur_display]))
    def Seg(self):
        self.label_19.setText('Start Segmentation')
        app.processEvents()
        
        start = time.process_time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = UNet(n_channels=1, n_classes=1)
        net.to(device=device)
        net.load_state_dict(torch.load(Uwei, map_location=device))
        net.eval()
        k1=0
        for file in save_all_fillname:
            self.label_19.setText('Segmentaion'+ file +' progress: '+str(k1+1)+'/'+str(total_imgc))
            app.processEvents()
            k1=k1+1
            lines=0
            with open(lebelpath+file+'.txt', 'r') as f:
                data = f.readlines()  
                for i in range(len(data)): 
                    data[i]=list(map(float,filter(None,re.split('[\t \n]',data[i].strip()))))
            lines=len(data)
            oriimag=cv2.imread(oriimagepath+file+'.png') 
            orimask=cv2.imread(orimaskpath+file+'.png',0)
            ret,orimask=cv2.threshold(orimask,8,255,cv2.THRESH_BINARY)
            imah,imaw,_ = oriimag.shape
            distmask=numpy.full((imah,imaw,3), (0,0,0), dtype=numpy.uint8)
            distmasksingle=numpy.full((imah,imaw), (0), dtype=numpy.uint8)
            for i in range(lines):
                rota=0##旋轉?
                # print(file)
                classs=int(data[i][0])
                xcenter=data[i][1]
                ycenter=data[i][2]
                yolow=data[i][3]
                yoloh=data[i][4]
                xmax=int(0.5*(yolow*imaw+xcenter*imaw*2))
                xmin=int(abs(yolow*imaw-xmax))
                ymax=int(0.5*(yoloh*imah+ycenter*imah*2))
                ymin=int(abs(yoloh*imah-ymax))
                if(classs==2):
                    distmasksingle[ymin:ymax,xmin:xmax]=255
                    colorcropmask=numpy.full((abs(ymax-ymin),abs(xmax-xmin),3), (255,0,0), dtype=numpy.uint8)
                    distmask[ymin:ymax,xmin:xmax]=(numpy.array(distmask[ymin:ymax,xmin:xmax]).astype(numpy.uint8)|colorcropmask)
                    continue
                cropped_image = oriimag[ymin:ymax,xmin:xmax]
                cropped_image=cv2.cvtColor(cropped_image,cv2.COLOR_RGB2GRAY)
                crope_hi,crope_wi=cropped_image.shape
                if((abs(xmax-xmin)>(imaw*0.8))&(abs(xmax-xmin)>(abs(ymax-ymin)*0.8))&(classs==1)):
                    colorcropmask=numpy.full((abs(ymax-ymin),abs(xmax-xmin),3), (0,255,0), dtype=numpy.uint8)
                    distmask[ymin:ymax,xmin:xmax]=(numpy.array(distmask[ymin:ymax,xmin:xmax]).astype(numpy.uint8)|colorcropmask)
                    distmasksingle[ymin:ymax,xmin:xmax]=255
                    continue

                if(crope_wi>=crope_hi):
                    cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
                    rota=1
                if(classs==1):
                    cropped_image=~cropped_image
                cropped_imageb=cv2.GaussianBlur(cropped_image,(0,0),5)
                cropped_image=cv2.addWeighted(cropped_image, 1.5, cropped_imageb, -0.5, 0)
                cropped_image=cv2.resize(cropped_image,(TAR_WIDTH,TAR_HEIGHT),interpolation = cv2.INTER_CUBIC)
                cropped_image = cropped_image.reshape(1, 1, cropped_image.shape[0], cropped_image.shape[1])
                img_tensor = torch.from_numpy(cropped_image)
                img_tensor = img_tensor.to(device=device, dtype=torch.float32)
                pred = net(img_tensor)
                pred = numpy.array(pred.data.cpu()[0])[0]
                pred[pred >= 0.5] = 255
                pred[pred < 0.5] = 0
                # ####pred預測的小圖的單通道mask
                if(rota==1):
                    pred= cv2.rotate(pred, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if(classs==0):
                    pred=cv2.resize(pred,(crope_wi,crope_hi),interpolation = cv2.INTER_CUBIC)
                    distmasksingle[ymin:ymax,xmin:xmax]=(numpy.array(distmasksingle[ymin:ymax,xmin:xmax]).astype(numpy.uint8)|numpy.array(pred).astype(numpy.uint8))
                    predcolor=numpy.stack((numpy.zeros_like(pred),numpy.zeros_like(pred),pred), axis=-1).astype(numpy.uint8)
                    distmask[ymin:ymax,xmin:xmax]=(numpy.array(distmask[ymin:ymax,xmin:xmax]).astype(numpy.uint8)|predcolor)
                if(classs==1):
                    pred=cv2.resize(pred,(crope_wi,crope_hi),interpolation = cv2.INTER_CUBIC)
                    distmasksingle[ymin:ymax,xmin:xmax]=(numpy.array(distmasksingle[ymin:ymax,xmin:xmax]).astype(numpy.uint8)|numpy.array(pred).astype(numpy.uint8))
                    predcolor=numpy.stack((numpy.zeros_like(pred),pred,numpy.zeros_like(pred)), axis=-1).astype(numpy.uint8)
                    distmask[ymin:ymax,xmin:xmax]=(numpy.array(distmask[ymin:ymax,xmin:xmax]).astype(numpy.uint8)|predcolor)

            distmasksingle=distmasksingle/255
            orimask=orimask/255
            x=numpy.sum(distmasksingle,dtype=numpy.uint32)
            y=numpy.sum(orimask,dtype=numpy.uint32)
            xandy=numpy.sum((numpy.array(distmasksingle).astype(numpy.uint8) & numpy.array(orimask).astype(numpy.uint8)),dtype=numpy.uint32)
            dice=float(2*xandy)/float((x+y))
            Dice.append(dice)
            oriimag=oriimag+0.75*distmask
            cv2.imwrite(savefinalmask+file+'.png',distmask)
            cv2.imwrite(savefinalimgwithmask+file+'.png',oriimag)
        end = time.process_time()
        self.label_12.setText("Segmente FPS: "+str(round(((end-start)/total_imgc),3)))
        
        scene = QtWidgets.QGraphicsScene()    
        scene.setSceneRect(0, 0, 399,349)
        imgs = QtGui.QPixmap(savefinalmask+save_all_fillname[cur_display]+'.png')        
        imgs = imgs.scaled(399,349)
        scene.addPixmap(imgs)
        self.graphicsView_3.setScene(scene)
        scene = QtWidgets.QGraphicsScene()    
        scene.setSceneRect(0, 0, 449,399)
        imgs = QtGui.QPixmap(savefinalimgwithmask+save_all_fillname[cur_display]+'.png')        
        imgs = imgs.scaled(449,399)
        scene.addPixmap(imgs)
        self.graphicsView_4.setScene(scene)
        self.label_11.setText("Dice Coefficient: "+str(round(Dice[cur_display],3)))
        self.label_19.setText('Segmentation Done!')
        self.label_5.setText("Current Image : "+str(cur_display+1)+'/'+str(total_imgc))
        global segflag
        segflag=1
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = myMainWindow()
    window.show()
    sys.exit(app.exec_())