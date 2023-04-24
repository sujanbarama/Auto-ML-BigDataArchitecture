from tkinter import *
from results import *

class train(object):
    def __init__(self, main = None):
        self.root = main
        self.root.title('Train page')
        self.root.geometry('500x400')
        self.create()

    def create(self):
        self.f1 = Frame(self.root)
        self.f1.pack(pady=10)
        self.lab1 = Label(self.f1,text='What is your task?', font=('', 12)).pack(side="left")

        self.f4 = Frame(self.root)
        self.f4.pack(pady=10)
        Radiobutton(self.f4, text="Regression", value=0).pack(side="left")
        Radiobutton(self.f4, text="Classification", value=1).pack(side="left")

        self.f4 = Frame(self.root)
        self.f4.pack(pady=10)
        Button(self.f4,text="Train results",width=20, command=self.results).pack(side="left")


    def save(self):
        print("aa")

    def train(self):
        print("aa")




    def results(self):
        self.f1.destroy()
        self.f4.destroy()
        results(self.root)
