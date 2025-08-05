import cv2
import time
from picamera2 import Picamera2

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
 
model='apple1.tflite'
num_threads=4

dispW=640
dispH=360

picam2=Picamera2()
picam2.preview_configuration.main.size=(dispW,dispH)
picam2.preview_configuration.main.format='RGB888'
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

pos=(20,60)
font=cv2.FONT_HERSHEY_SIMPLEX
height=1
weight=3
myColor=(255,0,0)

labelHeight=1.5
labelColor=(255,0,0)
labelWeight=(2)

boxColor=(255,0,0)
boxWeight=2

#cent = (320,180)
rColor = (0,255,255)
cThick=5
r=0
 
fps=0
 
base_options=core.BaseOptions(file_name=model,use_coral=False, num_threads=num_threads)
detection_options=processor.DetectionOptions(max_results=1, score_threshold=.5)
options=vision.ObjectDetectorOptions(base_options=base_options,detection_options=detection_options)
detector=vision.ObjectDetector.create_from_options(options)
tStart=time.time()
while True:
    #ret, im = cam.read()
    frame=picam2.capture_array()
    frame=cv2.flip(frame,-1)
    cv2.putText(frame,str(int(fps))+' FPS',pos,font,height,myColor,weight)
    imRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    imTensor=vision.TensorImage.create_from_array(imRGB)
    myDetects=detector.detect(imTensor)
    print(myDetects)
    print()
    for myDetect in myDetects.detections:
        ###print(myDetect)
        #print(myDetect.bounding_box.origin_x,myDetect.bounding_box.origin_y)
        UL=(myDetect.bounding_box.origin_x,myDetect.bounding_box.origin_y)
        LR=(myDetect.bounding_box.origin_x+myDetect.bounding_box.width,myDetect.bounding_box.origin_y+myDetect.bounding_box.height)
        
        
        cent = ()
        for i in range(len(UL)):
          cent += (int(UL[i]/2) + int(LR[i]/2),)
          

        
        
        print(myDetect.categories[0].category_name)
        objName=myDetect.categories[0].category_name
        ###
        if objName=='manzana':
            frame=cv2.rectangle(frame,UL,LR,boxColor,boxWeight)       
            frame=cv2.circle(frame,cent,r,rColor,cThick)
            
            cv2.putText(frame,'manzana',UL,font,labelHeight,labelColor,labelWeight)
            

        ####
        #frame=cv2.rectangle(frame,UL,LR,boxColor,boxWeight)
        #cv2.putText(frame,objName,UL,font,labelHeight,labelColor,labelWeight)
        print(UL,LR)
        print(cent)
        print()
    #image=utils.visualize(frame, myDetects)
    
    
    
    cv2.imshow('frame', frame) 
    if cv2.waitKey(1) == ord('q'): 
        break
    tEnd=time.time()
    loopTime=tEnd-tStart
    fps= .9*fps +.1*1/loopTime
    print(fps)
    tStart=time.time()
#vid.release() 
cv2.destroyAllWindows()