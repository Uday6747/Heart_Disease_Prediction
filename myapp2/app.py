from flask import Flask,render_template,request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import warnings 
warnings.simplefilter('ignore')


app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train")
def train():
    global X_train, X_test, Y_train, Y_test
    global model
    heart_df = pd.read_csv('Heart_Disease_Prediction.csv')
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    encoder.fit(heart_df['Heart Disease'])
    heart_df['Heart Disease'] = encoder.transform(heart_df['Heart Disease'])
    X = heart_df.drop(columns ='Heart Disease', axis =1)
    Y = heart_df['Heart Disease']


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify = Y, random_state=2)


    model = LogisticRegression()
    model.fit(X_train,Y_train)

    return render_template("train.html")

@app.route("/accuracy")
def accuracy():
    X_train_prediction = model.predict(X_train)
    acc = accuracy_score(X_train_prediction, Y_train)
    return render_template("accuracy.html",facc=acc)

    
@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/prediction",methods=['POST'])
def prediction():
    a1=request.form['t1']
    a2=request.form['t2']
    a3=request.form['t3']
    a4=request.form['t4']
    a5=request.form['t5']
    a6=request.form['t6']
    a7=request.form['t7']
    a8=request.form['t8']
    a9=request.form['t9']
    a10=request.form['t10']
    a11=request.form['t11']
    a12=request.form['t12']
    a13=request.form['t13']
    predicted = model.predict([[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13]])

    return render_template("prediction.html",p=predicted)


app.run(debug=True)

