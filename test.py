import cv2,threading
from PIL import *
import pygame
from pygame.locals import *
from pygame import mixer 
import time
import os
import shutil
import glob
import RPi.GPIO as GPIO
import face_recognition as fr
from datetime import datetime

pygame.init()
WIDTH = 1280
HEIGHT = 1080
windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), FULLSCREEN, 31)
pictureHeight=500
pictureWidth=500
timer=0
index=1
lastPicture="face_0"
pictureIndex=1
status="idle"
currentDirectory='/home/pi/Desktop/ProjectFiles/pictures/current/'
oldDirectory='/home/pi/Desktop/ProjectFiles/pictures/old/'
firstDirectory='/home/pi/Desktop/ProjectFiles/pictures/first/'  
pumpRunning=False
first=True
GPIO.setmode(GPIO.BCM) 
GPIO.setwarnings(False) 
pins = 17  
GPIO.setup(pins, GPIO.OUT)
mixer.init()
pygame.mixer.Channel(0).play(pygame.mixer.Sound('/home/pi/Desktop/ProjectFiles/Ambient-v2.wav'),-1)
pygame.mixer.Channel(1).set_volume(0.5)

def backup():
    global currentDirectory,oldDirectory
    files = os.listdir(currentDirectory)
    for file in files:
        new_path = shutil.move(f"{currentDirectory}/{file}", '/home/pi/Desktop/ProjectFiles/pictures/backup/')
        time=datetime.now()
        os.rename(r'/home/pi/Desktop/ProjectFiles/pictures/backup/'+str(file),r'/home/pi/Desktop/ProjectFiles/pictures/backup/'+str(time))

def movePictures():
    global currentDirectory,oldDirectory
    files = os.listdir(currentDirectory)
    for file in files:
        shutil.move(f"{currentDirectory}/{file}", oldDirectory)

def deletePictures(index):
    global currentDirectory,oldDirectory
    if(index=='all'):
        files = glob.glob(oldDirectory+"*")
        for f in files:
            os.remove(f)
        files = glob.glob(currentDirectory+"*")
        for f in files:
            os.remove(f)
    elif(index=="old"):
        files = glob.glob(oldDirectory+"*")
        for f in files:
            os.remove(f)


def takePicture(img):
    global lastPicture,pictureWidth,pictureHeight,pictureIndex
    splitName=lastPicture.split("_")
    #print(splitName)
    #index=int(splitName[1])+1
    
    # save image
    lastPicture=splitName[0]+"_"+str(pictureIndex)
    dim = (pictureWidth, pictureHeight)

    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    try:
        cv2.imwrite('/home/pi/Desktop/ProjectFiles/pictures/current/'+lastPicture+'.bmp',resized)
        pictureIndex+=1
    except:
        print("Error has occured")
    #print("Image written to file-system : ",status)

def displayImages():
    global status,windowSurface,currentDirectory,firstDirectory,oldDirectory,index,timer,first
    mainLoop = True
    directory=""
    index=1
    
    while mainLoop:
            
        
            
        if not os.listdir(oldDirectory):
            if(first):
                directory=firstDirectory
                first=False
        else:
            directory=oldDirectory
        if(index==len(os.listdir(directory))):
            index=1
        try:
            #print(directory)
            image = pygame.image.load(directory+'face_'+str(index)+'.bmp').convert()
            image = pygame.transform.scale(image, (800, 1080))
            #image.fill((30, 70, 30, 100), special_flags=pygame.BLEND_ADD)
            image.fill((0, 80, 0, 20), special_flags=pygame.BLEND_ADD)
            image_rect = image.get_rect(center = windowSurface.get_rect().center)
            
            

            windowSurface.blit(image,image_rect)
            pygame.display.update()
            index+=1
            time.sleep(0.1)
        except:
            print("Image does not exist")
    print("Stopping display")        

imageDisplayThread = threading.Thread(target=displayImages)

def addTimer():
    global timer,status,imageDisplayThread,index,pictureIndex,GPIO,pins,sound,currentDirectory
    while(True):
        if(timer==3):
            pygame.mixer.Channel(1).fadeout(3000)
            GPIO.output(pins,True)
        if(timer==6):
            timer=0
            status="displaying"
            index=1
            pictureIndex=1
            if(len(os.listdir(currentDirectory))>0):
                deletePictures("old")    
                movePictures()
            
        else:
            time.sleep(1)
            timer+=1
            print("Timer: "+str(timer)+" s")

def doubleCheckFace(img):
    #Double check if detected face is actual face
    print("Double checking...")

backup()
deletePictures("all")
startTimer = threading.Thread(target=addTimer)
face_cascade = cv2.CascadeClassifier('/home/pi/Desktop/ProjectFiles/haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
startTimer.start()
imageDisplayThread.start()
while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face+
    faceIndex=1
    for (x, y, w, h) in faces:
        #STart pump
        if(w>150):
            timer=0
            faces = img[y:y + h, x:x + w]
            takePicture(faces)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
            if(pygame.mixer.Channel(1).get_busy()==False):
                pygame.mixer.Channel(1).play(pygame.mixer.Sound('/home/pi/Desktop/ProjectFiles/bubble-v2-short.wav'))

            if(pumpRunning==False):
                GPIO.output(pins, False) 
            #print(["Width= "+str(w),"Height="+str(h),"Pos x= "+str(x),"Pos y= "+str(y)])
            
        #else:
            #print("Too far")
        #print("Size of rectangle: " +str(w)+","+str(h))
    
    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        takePicture(img)
        #break
        
# Release the VideoCapture object
cap.release()





