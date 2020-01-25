import cv2
from tkinter import *
import tkinter.ttk as ttk
from time import sleep
from PIL import ImageTk, Image
import PIL
import numpy as np
import threading

def main():
    win = Tk()
    win.title("TwiSound")
    frame = Frame(win)
    frame.pack()

    webcam = cv2.VideoCapture(0)

    img_panel = Label(frame)
    img_panel.pack()

    stopEvent = threading.Event()

    def videoLoop():
        while not stopEvent.is_set():
            check, img = webcam.read()
            if not check:
                break
            img_tk = ImageTk.PhotoImage(PIL.Image.fromarray(img))
            img_panel.configure(image=img_tk)
            sleep(1)

    thread = threading.Thread(target=videoLoop, args=())
    thread.start()

    win.mainloop()

    webcam.release()
    stopEvent.set()
    

main()
