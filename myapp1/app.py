from flask import Flask,render_template,request

import numpy as np 
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor


app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train")
def train():
    global X_train, X_test, Y_train, Y_test
    global model
    dataset = shuffle(np.array(pd.read_csv("C:\\Users\\91903\\Desktop\\ML\\breast Cancer\\dataset.csv",header=1)))

    #data frame

    data_frame = pd.read_csv("C:\\Users\\91903\\Desktop\\ML\\breast Cancer\\dataset.csv",header=1)
    data_frame.drop(data_frame.columns[[0]], axis=1, inplace=True)
    dataset = shuffle(np.array(data_frame))
    extracted_dataset= []
    target = []

    #extract target column
    for row in dataset:
        extracted_dataset.append(row[1:])
        if row[0] == 'B':
            target.append(0)
        else:
            target.append(1)


    X_train, X_test, Y_train, Y_test= train_test_split(extracted_dataset,target,test_size=0.3)


    model = DecisionTreeClassifier(criterion = "entropy", max_depth = 50)
    model.fit(X_train,Y_train)

    return render_template("train.html")

@app.route("/accuracy")
def accuracy():
    predicted = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(Y_test,predicted,normalize=True)
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
    a14=request.form['t14']
    a15=request.form['t15']
    a16=request.form['t16']
    a17=request.form['t17']
    a18=request.form['t18']
    a19=request.form['t19']
    a20=request.form['t20']
    a21=request.form['t21']
    a22=request.form['t22']
    a23=request.form['t23']
    a24=request.form['t24']
    a25=request.form['t25']
    a26=request.form['t26']
    a27=request.form['t27']
    a28=request.form['t28']
    a29=request.form['t29']
    a30=request.form['t30']
    predicted = model.predict([[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30]])

    return render_template("prediction.html",p=predicted)


app.run(debug=True)

