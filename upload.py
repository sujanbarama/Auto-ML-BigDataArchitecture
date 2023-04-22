import os
import io
from google.cloud import storage
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
#from script import process_csv
from flask import Flask
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\sujan\git learning\git\bigdata\maximal-record-384001-406302dda581.json' 
def cloud_read(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data))
    print(f'Pulled down file from bucket {bucket_name}, file name: {blob_name}')
    return df
    # with blob.open('r') as f:
    #     csv_file = f.read()
    # return csv_file

def cloud_write(bucket_name, blob_name, csv_file):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(csv_file)
#list all blobs in gcs
def list_blobs(bucket_name):
    # bucket_name = "your-bucket-name"
    storage_client = storage.Client()
    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)
    # Note: The call returns a response only when the iterator is consumed.
    ll = []
    for blob in blobs:
        if blob.name.endswith('.csv'):
            ll.append(blob.name)
    return ll

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/pop_up', methods=['GET', 'POST'])
def pop_up():
    return render_template('pop_up.html')

@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    csv_files = list_blobs('automl-bigdata')
    return render_template('preprocess.html', data = csv_files)

@app.route('/training', methods=['GET', 'POST'])
def training():
    csv_files = list_blobs('automl-bigdata')
    return render_template('training.html', data = csv_files)


def normalize_and_encode(imputed_data):
    target = imputed_data[['Target']]
    imputed_data = imputed_data.drop('Target', axis=1)
    #normalizing numerical columns using robustscalar
    numerical_columns  = [x for x in imputed_data.columns if imputed_data[x].dtype in ['int64', 'float64']]
    scalar = RobustScaler(quantile_range=(25,75))
    scaled = scalar.fit_transform(imputed_data[numerical_columns])
    scaled = pd.DataFrame(scaled)
    scaled.columns = imputed_data[numerical_columns].columns
    
    #dropping cat columns with more than 10 categories
    cat_cols = [x for x in imputed_data.columns if x not in numerical_columns]
    cat_cols_to_drop = []
    for col in cat_cols:
        if imputed_data[col].value_counts().count()>10:
            cat_cols_to_drop.append(col)
    data_for_enc = imputed_data.drop(numerical_columns,axis=1)
    data_for_enc.drop(cat_cols_to_drop,axis=1,inplace=True)

    #encoding categorical varialbles
    enc_data= pd.get_dummies(data_for_enc, columns=data_for_enc.columns)
    
    encoded_data = pd.concat([scaled, enc_data, target], axis=1)

    encoded_data.to_csv('upload_encoded.csv',index = False)
    return encoded_data
    
@app.route('/impute', methods=['GET', 'POST'])
def impute():

    # if request.method == 'POST':
    filename = request.form.get("name")
    data = cloud_read('automl-bigdata', filename)
    Target = data[['Target']]
    data = data.drop('Target', axis = 1)
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
    imputed_data = pd.concat([imputed_num, data_cat_imputed, Target], axis=1)
    imputed_data = normalize_and_encode(imputed_data)
    imputed_data.to_csv('upload_imputed.csv',index = False)
    cloud_write('automl-bigdata', f'{filename.split("raw")[0]}imputed.csv','upload_imputed.csv')
    #data = {"file":r"C:\sujan\git learning\git\bigdata\upload_imputed.csv"}
    return render_template('after_preprocess.html')

    #return render_template('preprocess.html', data = csv_files)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            new_filename = f'{filename.split(".")[0]}_raw.csv'
            save_location = os.path.join(r'C:\sujan\git learning\git\bigdata\input', new_filename)
            file.save(save_location)
            cloud_write('automl-bigdata', new_filename ,save_location)


            #output_file = process_csv(save_location)
            #return send_from_directory('output', output_file)
            #return redirect(url_for('download'))

    return render_template('upload.html')

# @app.route('/download')
# def download():
#     return render_template('download.html', files=os.listdir('output'))

# @app.route('/download/<filename>')
# def download_file(filename):
#     return send_from_directory('output', filename)


if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()