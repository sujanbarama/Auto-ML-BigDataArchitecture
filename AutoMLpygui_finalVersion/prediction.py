from tkinter import *
from presults import *
from tkinter import filedialog
from tkinter import messagebox
from google.cloud import storage
import pickle
import pandas as pd
import os

class prediction(object):
    def __init__(self, main = None):
        self.root = main
        self.root.title('Processing page')
        self.root.geometry('500x400')
        self.create()

    def create(self):
        def generate():
            self.f1.destroy()
            self.f2.destroy()
            self.f3.destroy()
            self.f4.destroy()
            self.f5.destroy()
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'D:\SchoolWorks\ATSL-5124/automl-bigdata-7c2859c8477a.json'
            bucket_name = 'automl-bigdataarch'
            blob_name = 'model.pkl'
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            pickle_in = blob.download_as_string()
            pickled_model = pickle.loads(pickle_in)
            print(f'Pulled down file from bucket {bucket_name}, file name: {blob_name}')
            print(self.select_path.get())
            df = pd.read_csv(self.select_path.get())
            preds = pickled_model.predict(df)
            pd.DataFrame({'id': range(len(preds)), 'Target': preds}).to_csv('predictions.csv', index = False)
            presults(self.root)
        self.f1 = Frame(self.root)
        self.f1.pack(pady=10)

        self.f2 = Frame(self.root)
        self.f2.pack(pady=10)
        self.select_path = StringVar()
        Label(self.f2, text="File Path：").pack(side="left")
        select = Entry(self.f2, textvariable = self.select_path)
        select.pack(side="left")
        Button(self.f2, text="choose file", command = self.selectFile).pack(side="left")

        self.f3 = Frame(self.root)
        self.f3.pack(pady=10)
        self.lab1 = Label(self.f3,text='Is your dataset clean? ', font=('', 12)).pack(side="left")

        self.f4 = Frame(self.root)
        self.f4.pack(pady=10)
        Radiobutton(self.f4, text="Yes", value=0).pack(side="left")
        Radiobutton(self.f4, text="No", value=1).pack(side="left")

        self.f5 = Frame(self.root)
        self.f5.pack(pady=10)
        Button(self.f5,text="Continue to results",width=20, command=generate).pack(side="left")

    def selectFile(self):
        selected_file_path = filedialog.askopenfilename()
        self.select_path.set(selected_file_path)

    def presult(self):
        self.f1.destroy()
        self.f2.destroy()
        self.f3.destroy()
        self.f4.destroy()
        self.f5.destroy()
        presults(self.root)
