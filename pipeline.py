from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import HashingEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import set_config

def model(estimator, df1):
    numlist = df1.select_dtypes(include=["int64", "float64"]).columns
    objlist = df1.select_dtypes(include="object").columns
    onehot_cat = ["category_id", "category_name", "address_city", "diet", "size", "storage_temp", "weekday", "quarter"]
    diff = [x for x in objlist if x not in onehot_cat]
    
    categorical_transformer = Pipeline(steps=[('imputer1', SimpleImputer(strategy='constant', fill_value='missing')),('encoder1', OneHotEncoder(drop='first'))])  
    categorical_transformer2 = Pipeline(steps=[('imputer2',  SimpleImputer(strategy='constant', fill_value='missing')),('encoder2', HashingEncoder())])
    numeric_transformer = Pipeline(steps=[('imputer3', SimpleImputer(strategy='most_frequent'))])
    
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numlist),('cat', categorical_transformer, onehot_cat),('cat2', categorical_transformer2, diff)])
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', estimator)])
    
    
    return clf