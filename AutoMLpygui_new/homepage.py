from tkinter import *
from up import *
from process import *

class homepage(object):
    def __init__(self, main = None):
        self.root = main
        self.root.title('Fancy Auto ML lading page')
        self.root.geometry('500x400')
        self.create()

    def create(self):
        self.f1 = Frame(self.root)
        self.f1.pack(pady=10)
        Button(self.f1,text="Upload",width=8, command=self.upload).pack(side="left", padx=3)
        Button(self.f1,text="Preprocess",width=12, command=self.preprocess).pack(side="left", padx=3)
        Button(self.f1,text="Train",width=8, command=self.train).pack(side="left", padx=3)
        Button(self.f1,text="Predict",width=12, command=self.predict).pack(side="left", padx=3)

    def upload(self):
        self.f1.destroy()
        up(self.root)

    def preprocess(self):
        self.f1.destroy()
        process(self.root)

    def train(self):
        print("new")

    def predict(self):
        print("new")