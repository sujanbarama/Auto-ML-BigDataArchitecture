from tkinter import *
from results import *
from modeling import Modeling
import os
from google.cloud import storage
import pandas as pd
import io
import prediction

class train(object):
    def __init__(self, main = None):
        self.root = main
        self.root.title('Train page')
        self.root.geometry('500x400')
        self.create()

    def create(self):
        def train_results():
            for i in listbox.curselection():
                file_name = listbox.get(i)
            print(file_name)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'D:\SchoolWorks\ATSL-5124/automl-bigdata-7c2859c8477a.json'
            def cloud_read(bucket_name, blob_name):
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                data = blob.download_as_bytes()
                df = pd.read_csv(io.BytesIO(data))
                print(f'Pulled down file from bucket {bucket_name}, file name: {blob_name}')
                return df
            file = cloud_read('automl-bigdataarch', file_name)
            if Var1.get():
                Modeling().classification(file)
            else:
                Modeling().regression(file)
            self.f1.destroy()
            self.f3.destroy()
            self.f4.destroy()
            prediction.prediction(self.root)

        self.f1 = Frame(self.root)
        self.f1.pack(pady=10)
        self.lab1 = Label(self.f1,text='What is your task?', font=('', 12)).pack(side="left")

        self.f3 = Frame(self.root)
        self.f3.pack(pady=10)
        Var1 = IntVar()
        val = Radiobutton(self.f3, text="Regression", value=0, variable = Var1)
        val.pack(side="left")
        val1 = Radiobutton(self.f3, text="Classification", value=1, variable = Var1)
        val1.pack(side="left")
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'D:\SchoolWorks\ATSL-5124/automl-bigdata-7c2859c8477a.json'
        storage_client = storage.Client()
        # Note: Client.list_blobs requires at least package version 1.17.0.
        blobs = storage_client.list_blobs('automl-bigdataarch')
        # Note: The call returns a response only when the iterator is consumed.
        ll = []
        for blob in blobs:
            if blob.name.endswith('.csv'):
                ll.append(blob.name)
        items = [i.split('/')[-1] for i in ll]
        strvar_items = StringVar(value=items)
        listbox = Listbox(self.f1, cursor='cross', listvariable=strvar_items)
        listbox.pack()
        self.f4 = Frame(self.root)
        self.f4.pack(pady=10)
        Button(self.f4,text="Train results",width=20, command=train_results).pack(side="left")


    def save(self):
        print("aa")

    def train(self):
        print("aa")

    def results(self):
        self.f1.destroy()
        self.f3.destroy()
        self.f4.destroy()
        results(self.root)
