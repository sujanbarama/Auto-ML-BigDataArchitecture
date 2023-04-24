from tkinter import *
from train import *

class next(object):
    def __init__(self, main = None):
        self.root = main
        self.root.title('Processing result page')
        self.root.geometry('500x400')
        self.create()

    def create(self):
        self.f1 = Frame(self.root)
        self.f1.pack(pady=10)
        self.lab1 = Label(self.f1,text='Your file is ready. What do you want to do next?', font=('', 12)).pack(side="left")

        self.f4 = Frame(self.root)
        self.f4.pack(pady=10)
        Button(self.f4,text="Save preprocessed",width=20,command=self.save).pack(side="left",padx=10)
        Button(self.f4,text="Train model",width=20,command=self.train).pack(side="left",padx=10)


    def save(self):
        print("aa")

    def train(self):
        self.f1.destroy()
        self.f4.destroy()
        train(self.root)
