from tkinter import *
from homepage import *
from process import *
from tkinter import filedialog
from tkinter import messagebox
import os
from google.cloud import storage

class up(object):
    def __init__(self, main = None):
        self.root = main
        self.root.title('File upload page')
        self.root.geometry('500x400')
        self.create()

    def create(self):
        self.f1 = Frame(self.root)
        self.f1.pack(pady=10)
        Button(self.f1,text="Homepage",width=12, command=self.goHome).pack(side="left")

        self.f2 = Frame(self.root)
        self.f2.pack(pady=10)
        self.select_path = StringVar()
        Label(self.f2, text="File Path：").pack(side="left")
        Entry(self.f2, textvariable=self.select_path).pack(side="left")
        Button(self.f2, text="choose file", command=self.selectFile).pack(side="left")

        self.f3 = Frame(self.root)
        self.f3.pack(pady=10)
        Button(self.f3,text="Upload",width=8, command=self.doUpload).pack(side="left")

        self.f4 = Frame(self.root)
        self.f4.pack(pady=10)
        Button(self.f4,text="Continue to processing",width=20, command=self.preprocess).pack(side="left")

    def selectFile(self):
        selected_file_path = filedialog.askopenfilename()
        self.select_path.set(selected_file_path)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'D:\SchoolWorks\ATSL-5124/automl-bigdata-7c2859c8477a.json'
        bucket_name = 'automl-bigdataarch'
        blob_name = selected_file_path.split('/')[-1]
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(selected_file_path)
        print(blob_name)

    def goHome(self):
        self.f1.destroy()
        self.f2.destroy()
        self.f3.destroy()
        self.f4.destroy()
        homepage(self.root)

    def preprocess(self):
        self.f1.destroy()
        self.f2.destroy()
        self.f3.destroy()
        self.f4.destroy()
        process(self.root)

    def doUpload(self):
        messagebox.showinfo(title= 'success', message='Dataset uploaded successfully')
