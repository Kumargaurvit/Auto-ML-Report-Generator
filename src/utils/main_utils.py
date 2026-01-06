import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, r2_score

# Helper Function to Remove Outliers
def remove_outliers(data: pd.DataFrame, columns=None)->pd.DataFrame:
    if columns is None:
       columns = data.select_dtypes(include=['number']).columns

    for column in columns:
        q1 = data[column].quantile(0.25) 
        q3 = data[column].quantile(0.75)

        iqr = q3 - q1

        lower_fence = q1 - 1.5 * iqr 
        higher_fence = q3 + 1.5 * iqr

        data = data[(data[column] >= lower_fence) & (data[column] <= higher_fence)]
    
    return data

# Helper Function to Check and Remove Null Values
def check_remove_null(data: pd.DataFrame)->pd.DataFrame:
    for column in data.columns:
        if data[column].isnull().sum() == 0:
            pass
        else:
            data[column].dropna(inplace=True)
    
    return data

# Helper Function to Check and Remove Duplicate Values
def check_remove_duplicates(data: pd.DataFrame)->pd.DataFrame:
    if data.duplicated().sum() > 0:
        data.drop_duplicates(inplace=True)
    else:
        pass

    return data

# Helper Function to Convert String columns to numerical
def encode_columns(X_train, X_test, y_train, y_test):
    # For independent feature
    for column in X_train.columns:
        if X_train[column].dtype == 'O':
                ohe = OneHotEncoder(handle_unknown='ignore')

                train_column_encoded = ohe.fit_transform(X_train[[column]]).toarray() 
                test_column_encoded = ohe.fit_transform(X_test[[column]]).toarray()

                train_column_encoded = pd.DataFrame(train_column_encoded,columns=ohe.get_feature_names_out(),index=X_train.index)
                test_column_encoded = pd.DataFrame(test_column_encoded,columns=ohe.get_feature_names_out(),index=X_test.index)

                X_train.drop(column,axis=1,inplace=True)
                X_test.drop(column,axis=1,inplace=True)

                X_train = pd.concat([X_train,train_column_encoded],axis=1)
                X_test = pd.concat([X_test,test_column_encoded],axis=1)
    
    # For dependent feature
    if y_train.dtype == 'O':
        le = LabelEncoder()

        y_train = le.fit_transform(y_train.values.ravel())
        y_test = le.transform(y_test.values.ravel())

    return X_train, X_test, y_train, y_test

# Helper Function to Evaluate Classification Models
def evaluate_models_classification(X_train,X_test,y_train,y_test,models):
    classification_metrics = {}

    for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test,y_pred)

            classification_metrics[model_name] = accuracy
    
    return classification_metrics

# Helper Function to Evaluate Regression Models
def evaluate_models_regression(X_train,X_test,y_train,y_test,models):
    regression_metrics = {}

    for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test,y_pred)

            regression_metrics[model_name] = r2
    
    return regression_metrics