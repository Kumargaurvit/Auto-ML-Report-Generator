import streamlit as st
import pandas as pd
import sys

from src.utils.main_utils import remove_outliers, check_remove_null, check_remove_duplicates
from src.components.model_trainer import ModelTrainer
from src.exception.exception import MLException
from src.logging.logger import logging
from src.components.prompt import llm_answer

# Class Object initialization
tp = ModelTrainer()

st.set_page_config("Report",layout="wide")
st.title('Auto-ML Report Generator')

uploaded_file = st.file_uploader("Upload CSV file", type='csv')

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.header("Data Preview")
    st.dataframe(data)

    try:
        remove_columns = st.sidebar.multiselect("Drop Columns",options=data.columns.to_list())
        if remove_columns:
            logging.info("Removing Columns")
            data.drop(columns=remove_columns,inplace=True)
        else:
            pass
    except Exception as e:
        raise MLException(e,sys)

    try:
        outliers = st.sidebar.radio("Remove Outliers",options=["Yes","No"],index=1)
        if outliers == "Yes":
            logging.info("Removing Outliers")
            clean_data = remove_outliers(data)
        else:
            clean_data = data.copy()
    except Exception as e:
        raise MLException(e,sys)
    
    try:
        remove_null = st.sidebar.radio("Remove Null Values",options=["Yes","No"],index=1)
        if remove_null == "Yes":
            logging.info("Removing Null Values")
            clean_data = check_remove_null(clean_data)
        else:
            pass
    except Exception as e:
        raise MLException(e,sys)
    
    try:
        remove_duplicates = st.sidebar.radio("Remove Duplicate Values",options=["Yes","No"],index=1)
        if remove_duplicates == "Yes":
            logging.info("Removing Duplicate Values")
            clean_data = check_remove_duplicates(clean_data)
            st.header("Cleaned Data")
            st.dataframe(clean_data)
        else:
            pass
    except Exception as e:
        raise MLException(e,sys)
    
    try:
        target_column = st.sidebar.selectbox("Choose Target Column",options=clean_data.columns,index=None)
        data = clean_data.head(2)
        if target_column:
            logging.info("Splitting Data into Training and Testing")
            X_train, X_test, y_train, y_test = tp.train_test_split(clean_data,target_column)
            problem_type = st.sidebar.radio("Select Model",options=["Classification","Regression"],index=None)
            if problem_type == "Classification":
                with st.spinner("⏳ Training Models and Generating Report..."):
                    logging.info("Training Classification Models")
                    best_model_name, best_model_score, classification_metrics = tp.train_models_classification(X_train,X_test,y_train,y_test)
                    response = llm_answer(model_metrics=classification_metrics,data=data)
                st.success(f"Best Model Selected {best_model_name} : {best_model_score:.2f}")
                st.header("Report:")
                st.markdown(response)
                if response:
                    st.download_button(
                        "Download ML Report",
                        data = response,
                        file_name="ML Report.txt"
                    )
            elif problem_type == None:
                pass
            else:
                with st.spinner("⏳ Training Models and Generating Report..."):
                    logging.info("Training Regression Models")
                    best_model_name, best_model_score, regression_metrics = tp.train_models_regression(X_train,X_test,y_train,y_test)
                    response = llm_answer(model_metrics=regression_metrics,data=data)
                st.success(f"Best Model Selected {best_model_name} : {best_model_score:.2f}")
                st.header("Report:")
                st.markdown(response)
                if response:
                    st.download_button(
                        "Download ML Report",
                        data = response,
                        file_name="ML Report.txt"
                    )
        else:
            pass
    except Exception as e:
        raise MLException(e,sys)
    
else:
    st.info('Please Upload the CSV File for Report Generation')