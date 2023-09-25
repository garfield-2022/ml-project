# -*- coding: utf-8 -*-
"""
Created on Thu Sep 7 13:14:30 2023

@author: yingfan
"""

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("model.pkl","rb")
model=pickle.load(pickle_in)


@app.route('/predict',methods=["Get"])
def predict_class():
    
    """Predict if the transaction is Normal or Fraudulent.
    ---
    parameters:  
      - name: Time
        in: query
        type: float
        required: true
      - name: V1
        in: query
        type: float
        required: true
      - name: V2
        in: query
        type: float
        required: true
      - name: V3
        in: query
        type: float
        required: true  
      - name: V4
        in: query
        type: float
        required: true
      - name: V5
        in: query
        type: float
        required: true
      - name: V6
        in: query
        type: float
        required: true
      - name: V7
        in: query
        type: float
        required: true
      - name: V8
        in: query
        type: float
        required: true
      - name: V9
        in: query
        type: float
        required: true  
      - name: V10
        in: query
        type: float
        required: true
      - name: V11
        in: query
        type: float
        required: true
      - name: V12
        in: query
        type: float
        required: true 
      - name: V13
        in: query
        type: float
        required: true
      - name: V14
        in: query
        type: float
        required: true
      - name: V15
        in: query
        type: float
        required: true
      - name: V16
        in: query
        type: float
        required: true
      - name: V17
        in: query
        type: float
        required: true
      - name: V18
        in: query
        type: float
        required: true 
      - name: V19
        in: query
        type: float
        required: true
      - name: V20
        in: query
        type: float
        required: true
      - name: V21
        in: query
        type: float
        required: true
      - name: V22
        in: query
        type: float
        required: true
      - name: V23
        in: query
        type: float
        required: true
      - name: V24
        in: query
        type: float
        required: true
      - name: V25
        in: query
        type: float
        required: true
      - name: V26
        in: query
        type: float
        required: true
      - name: V27
        in: query
        type: float
        required: true
      - name: V28
        in: query
        type: float
        required: true 
      - name: Amount
        in: query
        type: float
        required: true   
      
    responses:
        500:
            description: Prediction
        
    """
    Time=float(request.args.get("Time"))
    V1=float(request.args.get("V1"))
    V2=float(request.args.get("V2"))
    V3=float(request.args.get("V3"))
    V4=float(request.args.get("V4"))
    V5=float(request.args.get("V5"))
    V6=float(request.args.get("V6"))
    V7=float(request.args.get("V7"))
    V8=float(request.args.get("V8"))
    V9=float(request.args.get("V9"))
    V10=float(request.args.get("V10"))
    V11=float(request.args.get("V11"))
    V12=float(request.args.get("V12"))
    V13=float(request.args.get("V13"))
    V14=float(request.args.get("V14"))
    V15=float(request.args.get("V15"))
    V16=float(request.args.get("V16"))
    V17=float(request.args.get("V17"))
    V18=float(request.args.get("V18"))
    V19=float(request.args.get("V19"))
    V20=float(request.args.get("V20"))
    V21=float(request.args.get("V21"))
    V22=float(request.args.get("V22"))
    V23=float(request.args.get("V23"))
    V24=float(request.args.get("V24"))
    V25=float(request.args.get("V25"))
    V26=float(request.args.get("V26"))
    V27=float(request.args.get("V27"))
    V28=float(request.args.get("V28"))
    Amount=float(request.args.get("Amount"))
    prediction=model.predict([[Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount]])
    print(prediction[0])
    return "Model prediction is"+str(prediction)

@app.route('/predict_file',methods=["POST"])
def prediction_test_file():
    """Prediction on multiple input test file .
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        500:
            description: Test file Prediction
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    prediction=model.predict(df_test)
    
    return str(list(prediction))

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')
    
 
# docker build . -f Dockerfile.txt -t mlops_app_docker

# docker container run -p 5000:5000 mlops_app_docker

# open a browser and type http://127.0.0.1:5000/apidocs/    
    
    
