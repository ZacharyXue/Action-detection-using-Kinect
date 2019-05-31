from tkinter import *
from tkinter.tix import Tk, Control, ComboBox  #升级的组合控件包
from tkinter.messagebox import showinfo, showwarning, showerror #各种类型的提示框
from PIL import Image, ImageTk

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
        self.change = Button(self.bottom_frame,text='Continue',command=self.change_func)
        self.change.grid(row=1,column=1)
        # 退出
        self.quit = Button(self.bottom_frame,text='  Quit  ',command=self.master.quit)
        self.quit.grid(row=1,column=2)

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


if __name__ == "__main__":
    # initialize Tk
    root = Tk() 
    root.title("Action Recognition")   
    root.geometry("640x550")    
    root.resizable(width=True, height=True) # 设置窗口是否可以变化长/宽，False不可变，True可变，默认为True
    root.tk.eval('package require Tix')  #引入升级包

    basic_desk(root)

    root.mainloop()