from tkinter import *
from prediction import *

class results(object):
    def __init__(self, main = None):
        self.root = main
        self.root.title('Processing page')
        self.root.geometry('500x400')
        self.create()

    def create(self):
        self.f1 = Frame(self.root)
        self.f1.pack(pady=10)

        self.f4 = Frame(self.root)
        self.f4.pack(pady=10)
        Button(self.f4,text="Continue to prediction",width=20, command=self.prediction).pack(side="left")


    def prediction(self):
        self.f1.destroy()
        self.f4.destroy()
        prediction(self.root)
