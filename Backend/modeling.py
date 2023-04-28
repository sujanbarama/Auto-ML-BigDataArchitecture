from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import string
import random
from sklearn.metrics import balanced_accuracy_score, f1_score
from scipy import stats
import warnings
import pickle
from google.cloud import storage
warnings.filterwarnings("ignore")
     


class Modeling:
    def regression(self, train_data):
        #Models being trained for regression
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
        #assigning target
        y_class = train_data[['Target']]
        
        #test train split
        X_train, X_val, y_train, y_val = train_test_split(train_data.drop('Target', axis=1), y_class, test_size=0.2, random_state=100)
        
        res = {}
        #modeling
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
        #best parameters for models
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

        
    # have to put this in the server file
        pickle.dump(clf[best_model], open('model.pkl', 'wb'))
        cloud_write('automl-bigdataarch', 'model.pkl', 'model.pkl')
        # db.collection(u'models').document(string_name).set(res)


        return best_model
    

    def classification(self, train_data):
        #classification models
        classifiers = [
        XGBClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        LogisticRegression(),
        DecisionTreeClassifier()
        ]

        #target variable 
        y_class = train_data[['Target']]
        #train test split
        X_train, X_val, y_train, y_val = train_test_split(train_data.drop('Target', axis=1), y_class, test_size=0.2, random_state=100)

        res = {}
        #modeling
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
        #modeling best parameters
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
            #will remove once added to server file
            # string_name = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k = N))

            # while string_name in db.collection(u'models').stream():
                # string_name = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k = N))

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
        def cloud_write(bucket_name, blob_name, csv_file):
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(csv_file)
        # will remove once added to server file
        pickle.dump(clf[best_model], open('model.pkl', 'wb'))
        cloud_write('automl-bigdataarch', 'model.pkl', 'model.pkl')
        # db.collection(u'models').document(string_name).set(res)
        return best_model, res
    

            

