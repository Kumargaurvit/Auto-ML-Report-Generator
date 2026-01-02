FROM python:3.13-slim
COPY . /report
WORKDIR /report
RUN pip install -r requirements.txt
CMD ["streamlit","run","app.py"]    