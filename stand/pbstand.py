import cv2
from tkinter import *
import tkinter.ttk as ttk
from time import sleep
from PIL import ImageTk, Image
import PIL
import numpy as np
import threading
import dlib

num_images = 10 # Number of images to blend
queue_size = 100 # Number of queued images to chose from
# set queue_size = num_images to always blend latest images

size = 300
target_triangle = np.float32([[130.0,120.0],[170.0,120.0],[150.0,160.0]])


def find_biggest_face(faces):
    mx = 0
    res = None
    for x in faces:
        if x.right()-x.left()>mx:
            mx=x.right()-x.left()
            res = x
    return res,mx


def affine_transform(img,left_eye,right_eye,mouth):
    tr = cv2.getAffineTransform(np.float32([left_eye,right_eye,mouth]), target_triangle)                                
    return cv2.warpAffine(img,tr,(size,size))

def tkimage(img):
    return ImageTk.PhotoImage(PIL.Image.fromarray(img))


def main():
    win = Tk()
    win.title("PeopleBlending")
    frame = Frame(win)
    frame.pack()

    img_panel = Label(frame)
    img_panel.pack()

    webcam = cv2.VideoCapture(0)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    stopEvent = threading.Event()

    def videoLoop():
        images = np.zeros((queue_size,size,size,3),dtype=np.float32)    
        while not stopEvent.is_set():
            check, img = webcam.read()
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            if not check:
                break
            #img_tk = ImageTk.PhotoImage(PIL.Image.fromarray(img))
            #img_panel.configure(image=img_tk)
            faces = detector(img,1)
            if len(faces)>0:
                bface,mxwid = find_biggest_face(faces)
                if mxwid>img.shape[2]*0.05:
                    lmarks = predictor(img,bface)
                    lefteye = ((lmarks.part(37).x+lmarks.part(40).x)/2,(lmarks.part(37).y+lmarks.part(40).y)/2)
                    riteeye = ((lmarks.part(43).x+lmarks.part(46).x)/2,(lmarks.part(43).y+lmarks.part(46).y)/2)
                    mouth = (lmarks.part(67).x,lmarks.part(67).y)
                    images=np.roll(images,1,axis=0)
                    images[0,:,:,:] = affine_transform(img,lefteye,riteeye,mouth)
                    if queue_size==num_images:
                        i = np.average(images,axis=0).astype(np.uint8)
                    else:
                        idx = np.random.choice(queue_size,num_images)
                        tmp = images[idx]
                        i = np.average(tmp,axis=0).astype(np.uint8)
                    i = cv2.resize(i,(1000,1000))
                    img_tk = ImageTk.PhotoImage(PIL.Image.fromarray(i))
                    img_panel.configure(image=img_tk)
            sleep(1)

    thread = threading.Thread(target=videoLoop, args=())
    thread.start()

    win.mainloop()

    webcam.release()
    stopEvent.set()
    

main()
