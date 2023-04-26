from tkinter import *
from next import *
from results import *
import up
from cleaning import Cleaning
from google.cloud import storage
import os
import pandas as pd
import io

class process(object):
    def __init__(self, main = None):
        self.root = main
        self.file_name = ''
        self.root.title('Processing page')
        self.root.geometry('500x400')
        self.create()

    def file_name(self):
        return self.file_name

    def create(self):
        def selected_item():
            # print(help(listbox))
            for i in listbox.curselection():
                file_name = listbox.get(i)
            self.f1.destroy()
            self.f3.destroy()
            self.f4.destroy()
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'D:\SchoolWorks\ATSL-5124/automl-bigdata-7c2859c8477a.json'
            bucket_name = 'automl-bigdataarch'
            blob_name = file_name
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            data = blob.download_as_bytes()
            df = pd.read_csv(io.BytesIO(data))
            df_clean = Cleaning()
            df = df_clean.impute(df)
            df = df_clean.normalize_and_encode(df)
            df.to_csv('imputed.csv', index = False)
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name + 'preprocessed_gui.csv')
            selected_file_path = r'D:\SchoolWorks\ATSL-5124/imputed.csv'
            blob.upload_from_filename(selected_file_path)
            self.file_name = file_name
            next(self.root)

        self.f1 = Frame(self.root)
        self.f1.pack(pady=10)
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

        self.f3 = Frame(self.root)
        self.f3.pack(pady=10)
        Button(self.f3,text="Submit",width=8, command = selected_item).pack(side="left")

        self.f4 = Frame(self.root)
        self.f4.pack(pady=10)
        Button(self.f4,text="Continue to results",width=20, command=self.results).pack(side="left")


        # self.f5 = Frame(self.root)
        # self.f5.pack(pady=10)
        # Button(self.f1,text="Back",width=10, command=self.goBack).pack(side="left")


    def doSubmit(self):
        self.f1.destroy()
        self.f3.destroy()
        self.f4.destroy()
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'D:\SchoolWorks\ATSL-5124/automl-bigdata-7c2859c8477a.json'
        bucket_name = 'automl-bigdataarch'
        blob_name = f'raw_datafiles/{blob_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket()
        blob = bucket.blob()
        data = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(data))
        df_clean = Cleaning()
        df = df_clean.impute(df)
        df = df_clean.normalize_and_encode(df)
        next(self.root)

    # def goBack(self):
    #     self.f1.destroy()
    #     self.f3.destroy()
    #     self.f4.destroy()
    #     self.f5.destroy()
    #     up(self.root)

    def results(self):
        self.f1.destroy()
        self.f3.destroy()
        self.f4.destroy()
        results(self.root)
