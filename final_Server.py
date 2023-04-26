import os
import io
from google.cloud import storage
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
# from datetime import datetime
from flask import Flask
import pandas as pd
import sys
sys.path.insert(0, r'C:\Users\arman\Documents\GitHub\Auto-ML-BigDataArchitecture\Backend')
from cleaning import Cleaning
from modeling import Modeling
import pickle

import warnings
warnings.filterwarnings('ignore')

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\arman\Downloads\automl-bigdata-7c2859c8477a.json"

def cloud_read(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data))
    print(f'Pulled down file from bucket {bucket_name}, file name: {blob_name}')
    return df

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

def list_blobs_preprocessed(bucket_name):
    # bucket_name = "your-bucket-name"
    storage_client = storage.Client()
    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)
    # Note: The call returns a response only when the iterator is consumed.
    # ll = []
    for blob in blobs:
        if blob.name.endswith('_preprocessed.csv'):
            ll = blob.name
    return ll

def connection():
  cred = credentials.Certificate(r'C:\Users\arman\Documents\GitHub\Auto-ML-BigDataArchitecture\auto-ml-af39c-firebase-adminsdk-37cmd-35f3911f5e.json')
  try:
    app = firebase_admin.initialize_app(cred)
  except:
    app = firebase_admin.initialize_app(cred, name = str(random.random()))
  return firestore.client()

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')
@app.route('/uploading', methods=['GET', 'POST'])
def uploading():
    return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
  print('asjbakjbsd')
  #if request.method == 'POST':
  print(request)
  file = request.files['file']
  if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      new_filename = f'{filename.split(".")[0]}.csv'
      save_location = os.path.join(r'C:\Users\arman\Documents\GitHub\Auto-ML-BigDataArchitecture\input', new_filename)
      file.save(save_location)
      cloud_write('automl-bigdataarch', new_filename ,save_location)
  csv_files = list_blobs('automl-bigdataarch')
  csv_files = [i.split('/')[-1] for i in csv_files]
  return render_template('preprocess.html', data = csv_files)



@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    csv_files = list_blobs('automl-bigdataarch')
    csv_files = [i.split('/')[-1] for i in csv_files]
    return render_template('preprocess.html', data = csv_files)

@app.route('/preprocessResults', methods=['GET', 'POST'])
def preprocessResults():
    return render_template('preprocessResults.html')

@app.route('/Trainbutton', methods=['GET', 'POST'])
def Trainbutton():
    return render_template('training.html')


cleaner = Cleaning()

@app.route('/impute', methods=['GET', 'POST'])
def preprocessing():

    # if request.method == 'POST':
    filename = request.form.get("files")
    print(filename)
    data = cloud_read('automl-bigdataarch', filename)
    print(data.columns)
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
    missing = [x for x in percent_missing if x > 0.0]
    missing_rows_raw = len(missing)
    
    #cleaning the dataframe here
    imputed_data =cleaner.impute(data)

    # table for imputed data
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
    
    imputed_data = cleaner.normalize_and_encode(imputed_data)
    

    imputed_data.to_csv('upload_imputed.csv',index = False)
    cloud_write('automl-bigdataarch', f'{filename.split(".")[0]}_preprocessed.csv','upload_imputed.csv')
    return render_template('preprocessResults.html', data = frontend_data)
    #return render_template('preprocess.html', data = csv_files)

#TRAINING
#Regression
trainer= Modeling()

@app.route('/Regression', methods=['GET', 'POST'])
def regressor():
  file_name = list_blobs_preprocessed("automl-bigdataarch")
  train_data =pd.read_csv("C:\\Users\\arman\\Documents\\GitHub\\Auto-ML-BigDataArchitecture\\input\\{}".format(file_name))
  
  best_model, res=trainer.regression(train_data)

  # pickle.dump(clf[best_model], open('model.pkl', 'wb'))
  # cloud_write('automl-bigdataarch', 'model.pkl', 'model.pkl')
  #db.collection(u'models').document(string_name).set(res)
  return render_template('results.html', data = res)

#Classification
@app.route('/Classification', methods=['GET', 'POST'])
def classifier():
  file_name = list_blobs_preprocessed("automl-bigdataarch")
  train_data =pd.read_csv("C:\\Users\\arman\\Documents\\GitHub\\Auto-ML-BigDataArchitecture\\input\\{}".format(file_name))
  
  best_model, res=trainer.classification(train_data)

  # pickle.dump(clf[best_model], open('model.pkl', 'wb'))
  # cloud_write('automl-bigdataarch', 'model.pkl', 'model.pkl')
  #db.collection(u'models').document(string_name).set(res)
  return render_template('results.html', data = res)

@app.route('/predictions', methods = ['GET', 'POST'])
def predictions():
   return render_template('predict.html')

@app.route('/predictionresults', methods = ['GET', 'POST'])
def prediction_results():
   
   test=cloud_read("automl-bigdataarch", 'titanic_test.csv')
   test=cleaner.impute(test)
   test=cleaner.normalize_and_encode(test)
   def cloud_read_pickle(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    pickle_in = blob.download_as_string()
    my_dictionary = pickle.loads(pickle_in)
    #pickled_model = pickle.load(open(data, 'rb'))
    print(f'Pulled down file from bucket {bucket_name}, file name: {blob_name}')
    return my_dictionary
   
   pickled_model=cloud_read_pickle('automl-bigdataarch','model.pkl')
   pred_file = pickled_model.predict(test)
#    pd.DataFrame({'id': range(len(pred_file)), 'Predictions'})
   print(pred_file) 
   pd.DataFrame({"Predictions":pred_file}).to_csv("pred_file.csv",index=False)
   return render_template('predictResults.html')

if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()