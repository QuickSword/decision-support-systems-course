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

def model(estimator):
    numlist = df1.select_dtypes(include=["int64", "float64"]).columns
    objlist = df1.select_dtypes(include="object").columns
    onehot_cat = ["category_id", "category_name", "address_city", "diet", "size", "storage_temp", "weekday", "quarter"]
    diff = [x for x in objlist if x not in onehot_cat]
    
    
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\
                                              ('encoder', OneHotEncoder(drop='first'))])
    
    categorical_transformer2 = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\
                                               ('encoder', HashingEncoder())])
    
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))])
    
    
    preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numlist),('cat', categorical_transformer, onehot_cat),\
                  ('cat2', categorical_transformer2, diff)])
                  
    
    pipeline_est = make_pipeline(preprocessor, estimator)
    
    return pipeline_est