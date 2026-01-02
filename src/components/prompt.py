import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

def llm_answer(model_metrics,data):
    system_template = """
    You are a Skilled Report Generator which tells what model is best according to its metrics.
    All the metrics with model are provided below with model and its metrics
    {model_metrics} 

    And the dataset used is given below (Providing you 2 rows):
    {data}
    Now you have to generate a report telling why this model works best and also tell me all the models accuracy with their name 

    Give the Output in this format:
    Data Summary (explain each column with normal meaning not with technical words)
    All Model Explanation (Table and in table one column of meaning)
    Report why the model is best for this dataset (Do not show model table again)
    """

    prompt = PromptTemplate(template=system_template,input_variables=["model_metrics","data"])
    llm = ChatGroq(model="llama-3.1-8b-instant",api_key=st.secrets["GROQ_API_KEY"])


    retrieval_chain = prompt | llm | StrOutputParser()

    response = retrieval_chain.invoke({"model_metrics":model_metrics,"data":data})

    return response