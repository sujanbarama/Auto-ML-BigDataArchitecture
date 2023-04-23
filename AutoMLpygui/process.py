from tkinter import *
from next import *
from results import *

class process(object):
    def __init__(self, main = None):
        self.root = main
        self.root.title('Processing page')
        self.root.geometry('500x400')
        self.create()

    def create(self):
        self.f1 = Frame(self.root)
        self.f1.pack(pady=10)
        items = ('apple', 'orange', 'pear', 'grape')
        strvar_items = StringVar(value=items)
        Listbox(self.f1, cursor='cross', listvariable=strvar_items, selectmode='multiple').pack()

        self.f3 = Frame(self.root)
        self.f3.pack(pady=10)
        Button(self.f3,text="Submit",width=8, command=self.doSubmit).pack(side="left")

        self.f4 = Frame(self.root)
        self.f4.pack(pady=10)
        Button(self.f4,text="Continue to results",width=20, command=self.results).pack(side="left")


    def doSubmit(self):
        self.f1.destroy()
        self.f3.destroy()
        self.f4.destroy()
        next(self.root)

    def results(self):
        self.f1.destroy()
        self.f3.destroy()
        self.f4.destroy()
        results(self.root)
