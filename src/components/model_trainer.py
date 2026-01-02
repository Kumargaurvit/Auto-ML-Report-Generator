import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.naive_bayes import GaussianNB

from src.utils.main_utils import evaluate_models_classification, evaluate_models_regression, encode_columns

class ModelTrainer:
    def __init__(self):
        pass

    def train_models_regression(self,X_train,X_test,y_train,y_test):
        regression_models = {
            'Linear Regression' : LinearRegression(),
            'Lasso Regression' : Lasso(),
            'Ridge Regression' : Ridge(),
            'ElasticNet Regression' : ElasticNet(),
            'Decision Tree Regressor' : DecisionTreeRegressor(),
            'Random Forest Regressor' : RandomForestRegressor(),
            'Gradient Boosting Regressor' : GradientBoostingRegressor(),
            'XGBoost Regressor' : XGBRegressor(),
            'CatBoost Regressor' : CatBoostRegressor(allow_writing_files=False)
        }

        regression_metrics: dict = evaluate_models_regression(
            X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models=regression_models
        )
        
        best_model_score = max(sorted(regression_metrics.values()))

        best_model_name = list(regression_metrics.keys())[list(regression_metrics.values()).index(best_model_score)]

        return best_model_name, best_model_score*100, regression_metrics

    def train_models_classification(self,X_train,X_test,y_train,y_test):
        classification_models = {
            'Logistic Regression' : LogisticRegression(),
            'Decision Tree Classifier' : DecisionTreeClassifier(),
            'Random Forest Classifier' : RandomForestClassifier(),
            'Naive Bayes' : GaussianNB(),
            'Gradient Boosting Classifier' : GradientBoostingClassifier(),
            'XGBoost Classifier' : XGBClassifier(),
            'CatBoost Classifier' : CatBoostClassifier(allow_writing_files=False)
        }

        classification_metrics: dict = evaluate_models_classification(
            X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models=classification_models
        )

        best_model_score = max(sorted(classification_metrics.values()))

        best_model_name = list(classification_metrics.keys())[list(classification_metrics.values()).index(best_model_score)]

        return best_model_name, best_model_score*100, classification_metrics

    def train_test_split(self,data,target_column):
        X = data.drop(target_column,axis=1)
        y = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        X_train, X_test, y_train, y_test = encode_columns(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)

        X_train, X_test = self.scale_data(X_train,X_test)

        return X_train, X_test, y_train, y_test
        

    def scale_data(self,X_train,X_test):
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test