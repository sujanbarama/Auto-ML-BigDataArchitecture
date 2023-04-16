from flask import Flask
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
import pickle


 
# Flask constructor takes the name of
# current module (_name_) as argument.
app = Flask(__name__)
 
# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
from google.cloud import storage
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\sujan\git learning\git\bigdata\maximal-record-384001-406302dda581.json' 
def cloud_access(bucket_name, blob_name, csv_file):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(csv_file)
#end point
@app.route('/impute_data')
def impute():
    data = pd.read_csv(r'C:\sujan\git learning\git\bigdata\Dummy Data HSS.csv')
    #checking missing values
    percent_missing = data.isnull().sum() * 100 / data.shape[0]
    #dropping columns if missing percentage is more than 30
    for i in range(len(data.columns)):
        if percent_missing[i] >30:
            data.drop(data.columns[i],axis=1,inplace=True)
    #getting numerical and categorical variables
    numerical_columns = [x for x in data.columns if data[x].dtype != 'object']
    data_num = data[numerical_columns]
    
    cat_columns = [x for x in data.columns if x not in numerical_columns]
    data_cat = data[cat_columns]
    
    #Imputing using KNN Imputer for numerical columns
    imputer = KNNImputer(n_neighbors=2)
    imputed_num = imputer.fit_transform(data_num)
    imputed_num = pd.DataFrame(imputed_num)
    imputed_num.columns=data_num.columns
    
    # most frequent imputation for categorical columns
    data_cat_imputed = data_cat.apply(lambda x: x.fillna(x.value_counts().index[0]))
    
    #concat the imputed dfs
    imputed_data = pd.concat([imputed_num, data_cat_imputed], axis=1)
    imputed_data.to_csv('upload_imputed.csv',index = False)
    cloud_access('automl-bigdata', 'remove-2.csv','upload_imputed.csv')
    imputed_data.to_pickle('bvbhj.pickle')
    #return imputed_data
    return {
        'message': 'Success',
        # 'file': pickle.load('bvbhj.pickle')
    }
 
# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()