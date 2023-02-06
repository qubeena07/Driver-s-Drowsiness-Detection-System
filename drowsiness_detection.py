import cv2
import os
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time

#to play the alarms sound
mixer.init()
sound = mixer.Sound('alarm.wav')

#importing of face and eye classifier
face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
lbl=['Close','Open']

#load of the model
model = load_model('drow_model.h5')
model.summary()
path = os.getcwd()

#open of the webcam for detection
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

#Opening duration of the camera
while(True):
    ret, frame = cap.read()

    #frame size of camera
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Start of detection of the face and eyes
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    # Determining the face position
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    #Determining the right eye position
    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(32,32))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(32,32,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        out = model.predict(r_eye)
        rpred = np.argmax(out,axis=1)

        #Prediction of right eye
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    #Determining the left eye position
    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(32,32))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(32,32,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = np.argmax(model.predict(l_eye),axis=1)

        #Prediction of left eye
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    # Overall prediction of drowsiness
    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    #when score is less than 0, warning doesn't come.
    if(score<0):
        score=0   
    cv2.putText(frame,'Level:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    # when score is greater than 10, warning comes and alarm starts.
    if(score>10):
        #when person is feeling sleepy so we beep the alarm
        try:
            sound.play()
            cv2.putText(frame,"Drowsy",(600,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        except:
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
