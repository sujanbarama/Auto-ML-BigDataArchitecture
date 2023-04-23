import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler



class Cleaning:
        
    def impute(self, data):
        
        Target = data[['Target']]
        data = data.drop('Target', axis = 1)
    
        # checking missing values
        percent_missing = data.isnull().sum() * 100 / data.shape[0]
        
        # dropping columns if missing percentage is more than 30
        for i in range(len(data.columns)):
            if percent_missing[i] > 30:
                data.drop(data.columns[i], axis=1, inplace=True)
        
        # getting numerical and categorical variables
        numerical_columns = [x for x in data.columns if data[x].dtype != 'object']
        data_num = data[numerical_columns]
        
        cat_columns = [x for x in data.columns if x not in numerical_columns]
        data_cat = data[cat_columns]
        
        # Imputing using KNN Imputer for numerical columns
        imputer = KNNImputer(n_neighbors=2)
        imputed_num = imputer.fit_transform(data_num)
        imputed_num = pd.DataFrame(imputed_num)
        imputed_num.columns = data_num.columns
        
        # most frequent imputation for categorical columns
        data_cat_imputed = data_cat.apply(lambda x: x.fillna(x.value_counts().index[0]))
        
        # concat the imputed dfs
        imputed_data = pd.concat([imputed_num, data_cat_imputed, Target], axis=1)
        
        # return imputed_data
        return imputed_data
    
    def normalize_and_encode(self, imputed_data):
        target = imputed_data[['Target']]
        imputed_data = imputed_data.drop('Target', axis=1)
        # normalizing numerical columns using robustscalar
        numerical_columns = [x for x in imputed_data.columns if imputed_data[x].dtype in ['int64', 'float64']]
        scalar = RobustScaler(quantile_range=(25, 75))
        scaled = scalar.fit_transform(imputed_data[numerical_columns])
        scaled = pd.DataFrame(scaled)
        scaled.columns = imputed_data[numerical_columns].columns
        
        # dropping cat columns with more than 10 categories
        cat_cols = [x for x in imputed_data.columns if x not in numerical_columns]
        cat_cols_to_drop = []
        for col in cat_cols:
            if imputed_data[col].value_counts().count() > 10:
                cat_cols_to_drop.append(col)
        data_for_enc = imputed_data.drop(numerical_columns, axis=1)
        data_for_enc.drop(cat_cols_to_drop, axis=1, inplace=True)
        
        # encoding categorical variables
        enc_data= pd.get_dummies(data_for_enc, columns=data_for_enc.columns)
    
        encoded_data = pd.concat([scaled, enc_data, target], axis=1)

        encoded_data.to_csv('upload_encoded.csv',index = False)
        return encoded_data
    

    
