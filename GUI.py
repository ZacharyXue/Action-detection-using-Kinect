from tkinter import *
from tkinter.tix import Tk, Control, ComboBox  #升级的组合控件包
from tkinter.messagebox import showinfo, showwarning, showerror #各种类型的提示框
from PIL import Image, ImageTk

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import cv2 

class basic_desk():
    def __init__(self,master):
        self.master = master
        
        # 初始化进入界面
        self.basic = Frame(self.master,width=1000,height=1000)
        self.basic.pack()
        # 标题
        Label(self.basic,text='Action Recognition',font=("Arial",15)).pack()
        

        
        self.model_frame = Frame(self.basic)
        self.model_frame.pack()
        # Choose Neural Model
        # Create label
        model_label = Label(self.model_frame,text='The neural model: ')
        model_label.grid(row=1,column=0,rowspan=2,columnspan=2)
        
        self.model_type = StringVar()
        self.model_type.set('LSTM')
        model_LSTM = Radiobutton(self.model_frame,text='LSTM',variable=self.model_type,value='LSTM')
        model_LSTM.grid(row=5,column=1)
        model_CNN = Radiobutton(self.model_frame,text='CNN',variable=self.model_type,value='CNN')
        model_CNN.grid(row=5,column=6)

        self.skeleton_frame = Frame(self.basic)
        self.skeleton_frame.pack()
        # Choose the skeleton algorithm 
        # Create label
        skeleton_label = Label(self.skeleton_frame,text='The skeleton algorithm: ')
        skeleton_label.grid(row=1,column=0,rowspan=2,columnspan=2)
        
        self.skeleton_type = StringVar()
        self.skeleton_type.set('Kinect')
        skeleton_Kinect = Radiobutton(self.skeleton_frame,text='Kinect',variable=self.skeleton_type,value='Kinect')
        skeleton_Kinect.grid(row=5,column=1)
        skeleton_openpose = Radiobutton(self.skeleton_frame,text='OpenPose',variable=self.skeleton_type,value='OpenPose')
        skeleton_openpose.grid(row=5,column=6)

        # 底栏Frame
        self.bottom_frame = Frame(self.master)
        self.bottom_frame.pack(side=BOTTOM,anchor=SW)
        # 进入下一界面
        change = Button(self.bottom_frame,text='Continue',command=self.change_func)
        change.grid(row=1,column=1)
        # 退出
        _quit = Button(self.bottom_frame,text='  Quit  ',command=self.master.quit)
        _quit.grid(row=1,column=2)

    def change_func(self):
        # 进入下一界面
        self.basic.destroy()
        self.detect()
        # self.bottom_frame.destroy()
        # detect_desk(self.master,device=self.device,flag=flag)

    def detect(self):
        # 初始化进入界面
        self.detect_frame = Frame(self.master,width=1000,height=1000)
        self.detect_frame.pack()

        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body | PyKinectV2.FrameSourceTypes_Depth)

        # label info
        self.label = StringVar()

        if self.model_type == 'LSTM':
            pass
        elif self.model_type =='CNN':
            pass

        if self.skeleton_type == 'Kinect':
            pass
        elif self.skeleton_type == 'OpenPose':
            pass
        
        self.loop()

        # label info2
        action_label = Label(self.detect_frame,textvariable=self.label)
        action_label.pack(side=LEFT)
        
    def loop(self):
        # self.label.set("test")
        try:
            img = self._kinect.get_last_color_frame()
        except:
            success = False
        else:
            img = np.reshape(img,[1080,1920,4])
            self.img = cv2.cvtColor(img,cv2.COLOR_RGBA2RGB)
            success = True

        if success:
            
        self.detect_frame.after(1,self.loop)


if __name__ == "__main__":
    # initialize Tk
    root = Tk() 
    root.title("Action Recognition")   
    root.geometry("640x550")    
    root.resizable(width=True, height=True) # 设置窗口是否可以变化长/宽，False不可变，True可变，默认为True
    root.tk.eval('package require Tix')  #引入升级包

    basic_desk(root)

    root.mainloop()