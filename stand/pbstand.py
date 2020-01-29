import cv2
import os,datetime
from tkinter import *
import tkinter.ttk as ttk
from time import sleep
from PIL import ImageTk, Image
import PIL
import numpy as np
import threading
import dlib

disp_size = None # Size of the image to display on screen, or None for automatic 
num_images = 10 # Number of images to blend
queue_size = 100 # Number of queued images to chose from
# set queue_size = num_images to always blend latest images

queue_path = "c:/temp/pbstand/queue" # set to None to avoid writing
result_path = "c:/temp/pbstand/pics" # set to None to avoid
preload_queue = False # preload images from queue if exist 

# size and geometry of the image to manipulate
size = 300
target_triangle = np.float32([[130.0,120.0],[170.0,120.0],[150.0,160.0]])

face_proportion_threshold = 0.2 # Percentage of screen width that a face should occupy to be considered
sleep_interval = 1 # Number of seconds to sleep between image takes
redraw_period = 3 # Number of cycles to do between image redraw

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

descr_text="""
Этот экспонат строит портрет посетителей выставки в технике People Blending. Чтобы ваше лицо попало на портрет, 
задержитесь на некоторое время перед камерой. Камера делает по одному снимку в секунду, поэтому нескольких секунд
должно быть достаточно.
""".replace('\n',' ')

author_text="""
Автор: Дмитрий Сошников
Web: http://soshnikov.com
Instagram: @shwars
"""


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

def transform_image(img,bface):
    lmarks = predictor(img,bface)
    lefteye = ((lmarks.part(37).x+lmarks.part(40).x)/2,(lmarks.part(37).y+lmarks.part(40).y)/2)
    riteeye = ((lmarks.part(43).x+lmarks.part(46).x)/2,(lmarks.part(43).y+lmarks.part(46).y)/2)
    mouth = (lmarks.part(67).x,lmarks.part(67).y)
    return affine_transform(img,lefteye,riteeye,mouth)

def tkimage(img):
    return ImageTk.PhotoImage(PIL.Image.fromarray(img))

def create_queue():
    images = np.zeros((queue_size,size,size,3),dtype=np.float32)
    if queue_path is not None:
        os.makedirs(queue_path,exist_ok=True)
    if preload_queue and (queue_path is not None):
        print(" + Preloading queue")
        for i,fn in enumerate(sorted([os.path.join(queue_path,f) for f in os.listdir(queue_path)], key=os.path.getctime, reverse = True)[0:queue_size]):
            img = cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2RGB)
            faces = detector(img,1)
            if len(faces)>0:
                bface,mxwid = find_biggest_face(faces)
                images[i]=transform_image(img,bface)
    return images

def load_sample_image():
    x = cv2.imread("sample_image.jpg")
    x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
    x = cv2.resize(x,(disp_size,disp_size))
    tkx = ImageTk.PhotoImage(PIL.Image.fromarray(x))
    return tkx

def main():
    global disp_size
    win = Tk()
    win.title("PeopleBlending")
    win.geometry("{0}x{1}+0+0".format(win.winfo_screenwidth(), win.winfo_screenheight()))
    if disp_size is None:
        disp_size = win.winfo_screenheight()*9//10
    frame = Frame(win)
    frame.grid(row=0,column=0,sticky="nsew")

    frame.rowconfigure(0,weight=0)
    frame.rowconfigure(1,weight=10)
    frame.columnconfigure(0,weight=0)
    frame.columnconfigure(1,weight=1)

    img_panel = Label(frame)
    img_panel.grid(column=0,row=0,rowspan=3)

    title_label = Label(frame,text="People Blending",font=('Consolas',40))
    title_label.grid(column=1,row=0,sticky="nsew")

    descr = Label(frame,text=descr_text,font=('Consolas',18))
    # descr.bind('<Configure>', lambda e: descr.config(wraplength=descr.winfo_width()))
    descr.configure(wraplength=win.winfo_screenwidth()-disp_size-5)
    descr.grid(column=1,row=1,sticky="nsew")

    author = Label(frame,text=author_text,font=('Consolas',12))
    author.grid(column=1,row=2,sticky="nsew")

    webcam = cv2.VideoCapture(0)

    stopEvent = threading.Event()

    def videoLoop():
        nframe = 0
        img_panel.configure(image=load_sample_image())
        # descr.configure(text=descr_text)
        images = create_queue()    
        while not stopEvent.is_set():
            check, oimg = webcam.read()
            img = cv2.cvtColor(oimg,cv2.COLOR_BGR2RGB)
            if not check:
                break
            #img_tk = ImageTk.PhotoImage(PIL.Image.fromarray(img))
            #img_panel.configure(image=img_tk)
            faces = detector(img,1)
            if len(faces)>0:
                bface,mxwid = find_biggest_face(faces)
                if mxwid>img.shape[2]*face_proportion_threshold:
                    if queue_path is not None:
                        fn = datetime.datetime.now().strftime("pb-%Y-%m-%d-%H-%M-%S.jpg")
                        cv2.imwrite(os.path.join(queue_path,fn),oimg)
                    tr_img = transform_image(img,bface)
                    images=np.roll(images,1,axis=0)
                    images[0,:,:,:] = tr_img
            if nframe == 0:
                nframe = redraw_period
                if queue_size==num_images:
                    i = np.average(images,axis=0).astype(np.uint8)
                else:
                    idx = np.random.choice(queue_size,num_images)
                    tmp = images[idx]
                    i = np.average(tmp,axis=0).astype(np.uint8)
                if disp_size != size:
                    i = cv2.resize(i,(disp_size,disp_size))
                img_tk = ImageTk.PhotoImage(PIL.Image.fromarray(i))
                img_panel.configure(image=img_tk)
            sleep(sleep_interval)
            nframe-=1
        webcam.release()
        print("Goodbye! :)")

    thread = threading.Thread(target=videoLoop, args=())
    thread.start()

    win.mainloop()
    stopEvent.set()

    
main()
