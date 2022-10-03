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
from sklearn.ensemble import RandomForestClassifier
import joblib
from river import tree
from sklearn.preprocessing import LabelEncoder
import itertools
app = Flask(__name__)

@app.route('/create', methods = ['POST'])
def createModel():
    data = request.get_json()
    y = data['docType']
    dict(itertools.islice(data.items(), 1))
    data
    model = tree.HoeffdingTreeClassifier()
    model.learn_one(data, y)
    with open('decTree', 'wb') as picklefile:
        pickle.dump(model, picklefile)


    res = {"status": "success"}
    return make_response(jsonify(res), 200)

# app route for update
@app.route('/update', methods = ['POST'])
def update():
    update = request.get_json();
    y = update['docType']
    dict(itertools.islice(update.items(), 1))
    update
    model = pickle.load(open('decTree.model', 'rb'))
    model.learn_one(update, y)
    with open('decTree', 'wb') as picklefile:
        pickle.dump(model, picklefile)
    res = {"status": "success"}
    return make_response(jsonify(res), 200)



@app.route('/classify', methods = ['POST'])
def classify():
    data = request.get_json();
    txt = data['text']

    model = pickle.load(open('decTree.model', 'rb'))
    result_pred = model.predict_one(txt)

    res = {"result": result_pred}




    return make_response(jsonify(res), 200)





if __name__ == '__main__':
    app.debug = True
    app.run()


