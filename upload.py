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

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import string
import random
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pickle
from flask import Flask
from sklearn.metrics import balanced_accuracy_score, f1_score
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

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
  cred = credentials.Certificate('/content/auto-ml-af39c-firebase-adminsdk-37cmd-35f3911f5e.json')
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

@app.route('/upload', methods=['GET', 'POST'])
def upload():
  print('asjbakjbsd')
  #if request.method == 'POST':
  print(request)
  file = request.files['file']
  if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      new_filename = f'{filename.split(".")[0]}.csv'
      save_location = os.path.join(r'C:\sujan\git learning\git\bigdata\input', new_filename)
      file.save(save_location)
      cloud_write('automl-bigdataarch', new_filename ,save_location)
  csv_files = list_blobs('automl-bigdataarch')
  csv_files = [i.split('/')[-1] for i in csv_files]
  return render_template('preprocess.html', data = csv_files)

            #output_file = process_csv(save_location)
            #return send_from_directory('output', output_file)
            #return redirect(url_for('download'))
    # csv_files = list_blobs('automl-bigdataarch')
    # csv_files = [i.split('/')[-1] for i in csv_files]
    
    # return render_template('preprocess.html', data = csv_files)


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
    
    raw = 'C:\\sujan\\git learning\\git\\bigdata\\input\\'
    imputed_data.to_csv('{}{}_preprocessed.csv'.format(raw, filename.split('.')[0]),index = False)
    cloud_write('automl-bigdataarch', f'{filename.split(".")[0]}_preprocessed.csv','{}{}_preprocessed.csv'.format(raw, filename.split('.')[0]))

    return render_template('preprocessResults.html', data = frontend_data)

    #return render_template('preprocess.html', data = csv_files)

#TRAINING
#Regression
@app.route('/Regression', methods=['GET', 'POST'])
def regression():
  file_name = list_blobs_preprocessed("automl-bigdataarch")
  train_data =pd.read_csv("C:\\sujan\\git learning\\git\\bigdata\\input\\{}".format(file_name))
  reg_models = [
    KNeighborsRegressor(),
    LinearRegression(),
    GradientBoostingRegressor(),
    ExtraTreesRegressor(),
    RandomForestRegressor(),
    DecisionTreeRegressor(),
    Lasso(),
    Ridge()
]
  #db = connection()  
  y_class = train_data[['Target']]
  
  X_train, X_val, y_train, y_val = train_test_split(train_data.drop('Target', axis=1), y_class, test_size=0.2, random_state=100)
  
  res = {}
  
  KNeighborsRegressor_grid = {
      'n_neighbors':[2,5,10], 
      'weights': ['uniform', 'distance'], 
      'algorithm': ['auto','ball_tree','kd_tree','brute'],
      'leaf_size': [15,30,45],
      }

  GradientBoostingRegressor_grid = {
      'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
      'learning_rate':[0.1,0.5,0.8],
      'n_estimators':[10,50,100]
  }

  ExtraTreesRegressor_grid = {
      'n_estimators':[10,50,100],
      'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
  }

  RandomForestRegressor_grid = {
      'n_estimators':[10,50,100],
      'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
  }

  DecisionTreeRegressor_grid = {
      'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
      'splitter':['best','random']
  }

  LinearRegression_grid = {
    'fit_intercept': [True, False]
  }

  Lasso_grid = {
      'alpha': [0.1, 0.2, 0.5],
      'fit_intercept': [True, False]
  }
  Ridge_grid = {
       'alpha': [0.1, 0.2, 0.5],
      'fit_intercept': [True, False]
  }
  
 
  params = { 
      'KNeighborsRegressor': KNeighborsRegressor_grid,
      'GradientBoostingRegressor': GradientBoostingRegressor_grid,
      'ExtraTreesRegressor': ExtraTreesRegressor_grid,
      'RandomForestRegressor': RandomForestRegressor_grid,
      'DecisionTreeRegressor': DecisionTreeRegressor_grid,
      'LinearRegression': LinearRegression_grid, 
      'Lasso': Lasso_grid,
      'Ridge':Ridge_grid
    }

  clf = {}

  for reg in reg_models:
    name = reg.__class__.__name__  
    try:
      clf[name] = RandomizedSearchCV(reg, params[name], random_state=0)
    except:
      print(name)
      continue
    results = clf[name].fit(X_train, y_train)
    print(results.best_params_)
    r2 = round(r2_score(y_val, clf[name].predict(X_val)), 3)
    rmse = round(mean_squared_error(y_val, clf[name].predict(X_val)), 3)
    N = 16
 
    # string_name = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k = N))

    # while string_name in db.collection(u'models').stream():
    #     string_name = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k = N))

    print("{} trained with an RMSE of : {} and an accuracy of: {}".format(name, rmse, r2))
    
    res[name] = {
        'RMSE': rmse,
         'r2': r2,
         'params': results.best_params_
      }  

  rmse_list = []
  r2_list = []
  names = list(res.keys())
  for name in res:
    rmse_list.append(res[name]['RMSE'])
    r2_list.append(res[name]['r2'])

  if rmse_list.count(min(rmse_list)) > 1:
    best_model = names[r2_list.index(max(r2_list))]
  else:
    best_model = names[rmse_list.index(min(rmse_list))]

  print(best_model, clf[best_model].get_params())
  pickle.dump(clf[best_model], open('model.pkl', 'wb'))
  cloud_write('automl-bigdataarch', 'model.pkl', 'model.pkl')
  #db.collection(u'models').document(string_name).set(res)
  return render_template('results.html')

#Classification
@app.route('/Classification', methods=['GET', 'POST'])
def classification():
  file_name = list_blobs_preprocessed("automl-bigdataarch")
  train_data =pd.read_csv("C:\\sujan\\git learning\\git\\bigdata\\input\\{}".format(file_name))
  #db = connection()  
  classifiers = [
    XGBClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    LogisticRegression(),
    DecisionTreeClassifier()
    ]
  y_class = train_data[['Target']]
  X_train, X_val, y_train, y_val = train_test_split(train_data.drop('Target', axis=1), y_class, test_size=0.2, random_state=100)

  res = {}
  
  XGBClassifier_grid = {
      'n_estimators': stats.randint(50, 100),
      'learning_rate': stats.uniform(0.01, 0.59),
      'subsample': stats.uniform(0.3, 0.6),
      'max_depth': [3, 4, 5],
      'colsample_bytree': stats.uniform(0.5, 0.4),
      'min_child_weight': [1, 2, 3, 4]
      }

  RandomForestClassifier_grid = {
      'n_estimators':[10,50,100],
      'criterion':['gini', 'entropy', 'log_loss']
  }

  GradientBoostingClassifier_grid = {
      'loss':['log_loss', 'deviance', 'exponential'],
      'learning_rate':[0.1,0.5]
        }

  LogisticRegression_grid = {
    'penalty': ['l1', 'l2'],
    'dual':[True, False],
    'fit_intercept':[True,False]
  }

  DecisionTreeClassifier_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter':['best', 'random']
  }
  
  params = { 
      'XGBClassifier': XGBClassifier_grid,
      'RandomForestClassifier': RandomForestClassifier_grid,
      'GradientBoostingClassifier': GradientBoostingClassifier_grid,
      'LogisticRegression': LogisticRegression_grid,
      'DecisionTreeClassifier':DecisionTreeClassifier_grid
    }
    
  clf = {}
  
  for clf1 in classifiers:
    name = clf1.__class__.__name__
    try:
      clf[name] = RandomizedSearchCV(clf1, params[name], random_state=0)
    except:
        print(name)
        continue 

    results = clf[name].fit(X_train, y_train)
    print(results.best_params_)        
    acc = round(balanced_accuracy_score(y_val, clf[name].predict(X_val)), 3)
    f1 = round(f1_score(y_true=y_val, y_pred = clf[name].predict(X_val), average='weighted'), 3)

    N = 16

    # string_name = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k = N))

    # while string_name in db.collection(u'models').stream():
    #     string_name = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k = N))

    print("{} trained with an F1 of : {} and an accuracy of: {}".format(name, f1, acc))

    res[name] = {
        'Accuracy': acc,
        'F1Score': f1,
        'params': results.best_params_
      }  

  acc_list = []
  f1_list = []
  names = list(res.keys())
  for name in res:
    acc_list.append(res[name]['Accuracy'])
    f1_list.append(res[name]['F1Score'])

  if acc_list.count(max(acc_list)) > 1:
    best_model = names[f1_list.index(max(f1_list))]
  else:
    best_model = names[acc_list.index(max(acc_list))]

  print(best_model, clf[best_model].get_params())
  pickle.dump(clf[best_model], open('model.pkl', 'wb'))
  cloud_write('automl-bigdataarch', 'model.pkl', 'model.pkl')
  #db.collection(u'models').document(string_name).set(res)
  return render_template('results.html', data = res)


if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()