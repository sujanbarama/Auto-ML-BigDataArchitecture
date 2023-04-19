from flask import Flask
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
import pickle
from sklearn.model_selection import train_test_split
import io
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
import string
import random

 
# Flask constructor takes the name of
# current module (_name_) as argument.
app = Flask(__name__)
 
# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
from google.cloud import storage
import os

#to upload file to gcs
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\sujan\git learning\git\bigdata\maximal-record-384001-406302dda581.json' 
def cloud_write(bucket_name, blob_name, csv_file):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(csv_file)

# to read file from gcs
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


#end point for imputation
@app.route('/impute_data')
def impute():
    data = pd.read_csv(r'C:\sujan\git learning\git\bigdata\Dummy Data HSS.csv')
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
    imputed_data.to_csv('upload_imputed.csv',index = False)
    cloud_write('automl-bigdata', 'remove-2.csv','upload_imputed.csv')
    imputed_data.to_pickle('bvbhj.pickle')
    #return imputed_data
    return {
        'message': 'Success',
        # 'file': pickle.load('bvbhj.pickle')
    }

@app.route('/encoded')
def normalize_and_encode():
    imputed_data = cloud_read('automl-bigdata','remove-2.csv')
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

    cloud_write('automl-bigdata', 'encoded_data.csv','upload_encoded.csv')

    return {
        'message': 'Success',
        # 'file': pickle.load('bvbhj.pickle')
    }



#end point for training
@app.route('/train_data')
def training():
  reg_models = [
    KNeighborsRegressor(),
    GradientBoostingRegressor(),
    ExtraTreesRegressor(),
    RandomForestRegressor(),
    DecisionTreeRegressor(),
    LinearRegression(),
    Lasso(),
    Ridge()
]
  train_data = cloud_read('automl-bigdata','encoded_data.csv')
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

  for reg in reg_models:
    name = reg.__class__.__name__  
    try:
      clf = RandomizedSearchCV(reg, params[name], random_state=0)
    except:
      print(name)
      continue
    results = clf.fit(X_train, y_train)
    print(results.best_params_)
    r2 = round(r2_score(y_val, clf.predict(X_val)), 3)
    rmse = round(mean_squared_error(y_val, clf.predict(X_val)), 3)
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
    #db.collection(u'models').document(string_name).set(res) 
    return {
        'message': 'Success',
        # 'file': pickle.load('bvbhj.pickle')
    }
    


# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()