import cv2
import numpy as np
import cv2
import numpy as np
import os
import os
import cv2
import numpy as np
from PIL import Image
import time
from multiprocessing import Pool
from time import sleep
import multiprocessing
start = time.perf_counter()
def x1():
    print("yes")
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam=cv2.VideoCapture(0)
    id=3
    sampleNum=0
    while(True):
        print("yess")
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(img, 1.3, 5)
        for (x,y,w,h) in faces:
            sampleNum=sampleNum+1
            print (sampleNum)
            pth=r"dataset/User."
            cv2.imwrite("dataset/User."+str(id)+"."+str(sampleNum)+".jpg",img[y:y+h,x:x+h])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.imshow("Face",img)
        if(sampleNum>50):
                break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # print (sampleNum)
            break

    cam.release()
    cv2.destroyAllWindows()


def x2():
    cv2.face.LBPHFaceRecognizer_create()
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    path=r'dataset'
    def getImagesWithID(path):
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
        faces=[]
        IDs=[]
        for imagePath in imagePaths:
            faceImg=Image.open(imagePath).convert('L');
            faceNp=np.array(faceImg,'uint8')
            ID=int(os.path.split(imagePath)[-1].split('.')[1])
            faces.append(faceNp)
            # print(ID)
            IDs.append(ID)
            cv2.imshow("training",faceNp)
            cv2.waitKey(10)
        return IDs,faces

    Ids,faces = getImagesWithID(path)
    recognizer.train(faces,np.array(Ids))
    recognizer.save(r'recognizer/trainningData.yml')
    cv2.destroyAllWindows()


def redbull1():

    print("DONE!")
def redbull2():

    p=r'haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(p)
    cam=cv2.VideoCapture(0);
    recognizer=cv2.face.LBPHFaceRecognizer_create();
    recognizer.read(r'recognizer/trainningData.yml')
    id=0
    #font=cv2.FONT_HERSHEY_SIMPLEX,5,1,0,4
    font=cv2.FONT_HERSHEY_SIMPLEX
    while(True):
        ret, img = cam.read();
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(img, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            id,conf=recognizer.predict(gray[y:y+h,x:x+w])
            # print(id)
            if(id==1):
                id="Divik"
            if(id==2):
                id="Mukund"

            else:
                id="Divik"
            cv2.putText(img,str(id),(x,y+h),font,2,(255,255,255));                                                                        
            cv2.imshow('frame',img)
        if(cv2.waitKey(1) == ord('q')):
            break
    cam.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    pool = Pool(processes=24)
    a1 = pool.map_async(x1(), range(6))
    a2 = pool.map_async(x2(), range(6))
    a3 = pool.map_async(redbull2(), range(6))
    a4 = pool.map_async(redbull1(), range(6))
    finish = time.perf_counter()
    print(f' Finished in {round(finish-start , 2)} second(s)')