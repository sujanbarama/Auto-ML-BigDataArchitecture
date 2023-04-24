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

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\sujan\git learning\git\bigdata\automl-bigdata-7c2859c8477a.json' 
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

def connection():
  cred = credentials.Certificate('/content/auto-ml-af39c-firebase-adminsdk-37cmd-35f3911f5e.json')
  try:
    app = firebase_admin.initialize_app(cred)
  except:
    app = firebase_admin.initialize_app(cred, name = str(random.random()))
  return firestore.client()


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')
@app.route('/uploading', methods=['GET', 'POST'])
def uploading():
    return render_template('upload.html')

@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    csv_files = list_blobs('automl-bigdataarch')
    csv_files = [i.split('/')[-1] for i in csv_files]
    return render_template('preprocess.html', data = csv_files)

@app.route('/preprocessResults', methods=['GET', 'POST'])
def preprocessResults():
    return render_template('preprocessResults.html')

@app.route('/download/<path:filename>', methods=['GET'])
def download(filename):
    """Download a file."""
    
    full_path = os.path.join(app.root_path, r'C:\sujan\git learning\git\bigdata\input')
    
    return send_from_directory(full_path, 'orange_vs_grapefruit_raw.csv', as_attachment=True)



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
    try:
        enc_data= pd.get_dummies(data_for_enc, columns=data_for_enc.columns)
        encoded_data = pd.concat([scaled, enc_data, target], axis=1)
    except:
        encoded_data = scaled.copy()
    encoded_data.to_csv('upload_encoded.csv',index = False)
    return encoded_data
    
@app.route('/impute', methods=['GET', 'POST'])
def impute():

    # if request.method == 'POST':
    filename = request.form.get("files")
    print(filename)
    data = cloud_read('automl-bigdataarch', filename)
    print(data.columns)
    Target = data[['Target']]
    data = data.drop('Target', axis = 1)

    #data info
    # Dashboard for raw data
    num_cols_raw = [x for x in data.columns if data[x].dtype in ['int64', 'float64']]
    numerical_cols_raw = len(num_cols_raw)
    cat_cols_raw = [x for x in data.columns if data[x].dtype in ['object']]
    categorical_columns_raw = len(cat_cols_raw)
    num_columns_raw = [x for x in data.columns if data[x].dtype]
    number_columns_raw = len(num_columns_raw)
    num_rows_raw = len(data.index)

    #checking missing values
    percent_missing = data.isnull().sum() * 100 / data.shape[0]
    #dropping columns if missing percentage is more than 30
    for i in range(len(data.columns)):
        if percent_missing[i] >30:
            data.drop(data.columns[i],axis=1,inplace=True)
    missing = [x for x in percent_missing if x > 0.0]
    missing_rows_raw = len(missing)
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

    # Dashboard for imputed data
    num_cols_imp = [x for x in imputed_data.columns if imputed_data[x].dtype in ['int64', 'float64']]
    numerical_cols_imp = len(num_cols_imp)
    cat_cols_imp = [x for x in imputed_data.columns if imputed_data[x].dtype in ['object']]
    categorical_columns_imp = len(cat_cols_imp)
    num_columns_imp = [x for x in imputed_data.columns if imputed_data[x].dtype]
    number_columns_imp = len(num_columns_imp)
    num_rows_imp = len(imputed_data.index)
    missing_imp = [x for x in percent_missing if x > 0.30]
    missing_rows_imp = len(missing_imp)
    
    frontend_data = {
        'Raw_numericalvalues': numerical_cols_raw,
        'Raw_categoricalvalues': categorical_columns_raw,
        'Raw_columns': number_columns_raw,
        'Raw_rows': num_rows_raw,
        'Raw_missing': missing_rows_raw,
        'Imputed_numericalvalues': numerical_cols_imp,
        'Imputed_categoricalvalues': categorical_columns_imp,
        'Imputed_columns': number_columns_imp,
        'Imputed_rows': num_rows_imp,
        'Imputed_missingvalues': missing_rows_imp
      }
    imputed_data = normalize_and_encode(imputed_data)
    

    imputed_data.to_csv('upload_imputed.csv',index = False)
    cloud_write('automl-bigdataarch', f'{filename.split(".")[0]}_preprocessed.csv','upload_imputed.csv')
    #data = {"file":r"C:\sujan\git learning\git\bigdata\upload_imputed.csv"}
    # data = {'Raw_numericalvalues': 17,
    #             'Raw_categoricalvalues': 0,
    #             'Raw_columns': 17,
    #             'Raw_rows': 792,
    #             'Raw_missing': 0,
    #         'Imputed_numericalvalues': 17,
    #         'Imputed_categoricalvalues': 0,
    #         'Imputed_columns': 17,
    #         'Imputed_rows': 792,
    #         'Imputed_missingvalues': 0}
    return render_template('preprocessResults.html', data = frontend_data)

    #return render_template('preprocess.html', data = csv_files)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print(request)
        file = request.files['file']
        # print(len(file))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            new_filename = f'{filename.split(".")[0]}_raw.csv'
            save_location = os.path.join(r'C:\sujan\git learning\git\bigdata\input', new_filename)
            file.save(save_location)
            cloud_write('automl-bigdataarch', new_filename ,save_location)


            #output_file = process_csv(save_location)
            #return send_from_directory('output', output_file)
            #return redirect(url_for('download'))
    # csv_files = list_blobs('automl-bigdataarch')
    # csv_files = [i.split('/')[-1] for i in csv_files]
    
    # return render_template('preprocess.html', data = csv_files)

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