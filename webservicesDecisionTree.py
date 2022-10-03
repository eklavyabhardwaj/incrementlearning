import pickle

import pandas as pd
import river
from river.naive_bayes import MultinomialNB
from river.feature_extraction import BagOfWords, TFIDF
from river.compose import Pipeline
from river import metrics
from flask import Flask, render_template, request, Response, jsonify, make_response
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)

@app.route('/create', methods = ['POST'])
def createModel():
    data = request.get_json()
    import pandas as pd
    df = pd.DataFrame({"Text": data['text'], 'Doctype': data['docType']}, index= [0])
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['Text'] = le.fit_transform(df['Text'])
    X = df[['Text']]
    y = df[['Doctype']]
    model = DecisionTreeClassifier()
    model.fit(X,y)
    joblib.dump(model, 'decTree.model')


    res = {"status": "success"}
    return make_response(jsonify(res), 200)

# app route for update
@app.route('/update', methods = ['POST'])
def update():
    data = request.get_json();
    df_update= pd.DataFrame({"Text": data['text'], 'Doctype': data['docType']}, index = [0])
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_update['Text'] = le.fit_transform(df_update['Text'])
    X_update = df_update[['Text']]
    y_update = df_update[['Doctype']]
    model  = joblib.load(open('decTree.model', 'rb'))
    model.fit(X_update, y_update)

    joblib.dump(model, 'decTree.model')

    res = {"status": "success"}
    return make_response(jsonify(res), 200)



@app.route('/classify', methods = ['POST'])
def classify():
    data = request.get_json();
    df_update = pd.DataFrame(data,index = [0])
    le = LabelEncoder()
    df_update['text']=le.fit_transform(df_update['text'])
    X_pred = df_update[['text']]
    model  = joblib.load(open('decTree.model', 'rb'))
    result_pred =  model.predict(X_pred)
    res = {"result": result_pred}

    if result_pred == '0':
        res['result'] = 'CTSCAN'

    if result_pred == '1':
       res['result'] = 'DME-CMN-FORM'
    if result_pred == '2':
        res['result'] = 'Home Health'
    if result_pred == '3':
        res['result']="Hospital Consult"
    if result_pred == '4':
        res['result']="Hospital Progress Notes"
    if result_pred == '5':
        res['result']="Medication Correspondance"
    if result_pred == '6':
        res['result']="Office Note"
    if result_pred == '7':
        res['result']="PMSCPATIENT PAPER WORK"
    if result_pred == '8':
        res['result']="RECORD RELEASE"
    if result_pred == '9':
        res['result']="REFERRAL LETTERS"
    if result_pred == '10':
        res['result']="XRAY"
    if result_pred == '11':
        res['result']="test"



    return make_response(jsonify(res), 200)





if __name__ == '__main__':
    app.debug = True
    app.run()


