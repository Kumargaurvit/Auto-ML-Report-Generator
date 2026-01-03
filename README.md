# Auto-ML Report Generator 🚀

**Auto-ML Report Generator** is an intelligent Machine Learning automation tool that enables users to preprocess data, train multiple ML models, and generate detailed analytical reports — all through an interactive web interface.

The system provides **user-controlled preprocessing options** while fully automating the backend ML pipeline and **LLM-powered report generation using the Groq API via LangChain**.

---

## 🧠 Key Highlights

- Interactive UI for data preprocessing decisions
- Automated Machine Learning pipeline
- Best-model selection based on performance
- LLM-powered report generation using **Groq API**
- End-to-end workflow: Dataset → Model → Report

---

## 🎯 What This Project Does

The application allows users to make **data preprocessing choices** using buttons and select boxes, including:

- ✅ Outlier removal  
- ✅ Handling null values  
- ✅ Duplicate row removal  
- ✅ Target column selection  

Based on the **user’s selections**, the backend automatically:

1. Applies the chosen preprocessing steps  
2. Trains multiple Machine Learning models  
3. Evaluates model performance  
4. Selects the best-performing model  
5. Generates a **natural-language ML report** using an LLM  

---

## 🤖 LLM-Powered Report Generation (Groq API)

This project integrates a **Large Language Model (LLM)** using:

- **LangChain** for orchestration
- **Groq API** for ultra-fast LLM inference

### 🔹 How the LLM Works

- User preprocessing choices are captured from the UI
- Model training results and evaluation metrics are summarized
- LangChain constructs structured prompts
- The **Groq-hosted LLM** generates a detailed ML report explaining:
  - Data preprocessing decisions
  - Selected model and reasoning
  - Performance metrics
  - Key insights and conclusions

This approach eliminates the need for manually writing ML analysis reports while ensuring **clarity, speed, and consistency**.

---

## 🛠️ Tech Stack

### 🔹 Programming Language
- Python

### 🔹 Libraries & Frameworks
- **Pandas** – Data manipulation  
- **NumPy** – Numerical operations  
- **Scikit-learn** – Model training & evaluation  
- **Streamlit** – Interactive web UI  
- **LangChain** – LLM orchestration  
- **Groq API** – High-performance LLM inference  

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.9+
- pip
- Groq API Key

### Step 1: Clone Repository
```bash
git clone https://github.com/Kumargaurvit/Auto-ML-Report-Generator.git
cd Auto-ML-Report-Generator
```

### Step 2: Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Create .env File (Required)
Create a .env file in the root directory and add:
```env
LANGCHAIN_API_KEY=your_langchain_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### Step 4: Configure Streamlit Secrets (Required)
```toml
Create a .streamlit folder in the root directory and add a secrets.toml file:
GROQ_API_KEY="your_groq_api_key_here"
```