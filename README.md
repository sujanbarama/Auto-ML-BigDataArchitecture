# AUTOML

# README

## Backend

### Cleaning  class
This code includes a Python class named Cleaning that provides two methods for data cleaning: impute() and normalize_and_encode().

impute(data)
This method takes a Pandas DataFrame as input and performs missing value imputation. It checks for missing values in the input data, drops any columns with a missing percentage greater than 30%, and then separates the data into numerical and categorical variables.

The numerical variables are then imputed using the KNN Imputer from the sklearn.impute module, while the categorical variables are imputed using the most frequent value. Finally, the imputed numerical and categorical variables are concatenated back into a single DataFrame and returned.

normalize_and_encode(imputed_data)
This method takes the output from impute() method as input and performs normalization and categorical encoding. The numerical variables are normalized using the RobustScaler from the sklearn.preprocessing module. Categorical variables with more than 10 categories are dropped, and the remaining categorical variables are one-hot encoded using the get_dummies() method from the Pandas library. Finally, the normalized numerical and encoded categorical variables are concatenated back into a single DataFrame and returned.

Note that if there are no categorical variables to encode, the method will return the normalized numerical variables as is.

#### Example Usage

```

import pandas as pd
from Cleaning import Cleaning

# Load dataset
data = pd.read_csv('my_data.csv')

# Instantiate Cleaning class
cleaner = Cleaning()

# Impute missing values
imputed_data = cleaner.impute(data)

# Normalize and encode variables
cleaned_data = cleaner.normalize_and_encode(imputed_data)

# Save cleaned data to CSV
cleaned_data.to_csv('my_cleaned_data.csv', index=False)

```
