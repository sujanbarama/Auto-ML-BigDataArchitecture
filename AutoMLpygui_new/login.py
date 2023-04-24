from tkinter import *
from homepage import *

class login(object):
    def __init__(self, main = None):
        self.root = main
        self.root.title('User Login')
        self.root.geometry('350x200')
        self.create()

    def create(self):
        self.f1 = Frame(self.root)
        self.f1.pack(pady=10)
        self.lab1 = Label(self.f1,text='Username:', font=('', 14)).pack(side="left")
        self.ent1 = Entry(self.f1)
        self.ent1.pack(side="left")

        self.f2 = Frame(self.root)
        self.f2.pack(pady=10)
        self.lab2 = Label(self.f2,text='Password:', font=('', 14)).pack(side="left")
        self.ent2 = Entry(self.f2, show='*')
        self.ent2.pack(side="left")

        self.f3 = Frame(self.root)
        self.f3.pack(pady=10)
        Button(self.f3,text="Login",width=8, command=self.doLogin).pack(side="left")
        self.f4 = Frame(self.root)
        self.f4.pack(pady=10)
        Button(self.f4,text="New User",width=8,command=self.new).pack(side="left",padx=10)
        Button(self.f4,text="Forget Password",width=15,command=self.forget).pack(side="left",padx=10)

    def doLogin(self):
        self.f1.destroy()
        self.f2.destroy()
        self.f3.destroy()
        self.f4.destroy()
        homepage(self.root)

    def new():
        print("new")

    def forget():
        print("new")