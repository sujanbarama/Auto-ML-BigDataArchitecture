from tkinter import *

class presults(object):
    def __init__(self, main = None):
        self.root = main
        self.root.title('Processing page')
        self.root.geometry('500x400')
        self.create()

    def create(self):
        self.f1 = Frame(self.root)
        self.f1.pack(pady=10)
        self.lab1 = Label(self.f1,text='Your file is ready!', font=('', 12)).pack(side="left")
        Label(self.f1,text='The file is stored as predictions.csv!!!', font=('', 12)).pack(side="left")
        self.f4 = Frame(self.root)
        self.f4.pack(pady=10)
        # Button(self.f4,text="Save results",width=20, command=self.saveResult).pack(side="left")


    def saveResult(self):
        print("aa")
