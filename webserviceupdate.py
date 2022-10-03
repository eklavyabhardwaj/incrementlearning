import pickle

import pandas as pd
import river
from river.naive_bayes import MultinomialNB
from river.feature_extraction import BagOfWords, TFIDF
from river.compose import Pipeline
from river import metrics
from flask import Flask, render_template, request, Response, jsonify, make_response


app = Flask(__name__)

@app.route('/create', methods = ['POST'])
def createModel():
    data = request.get_json();
    txt = data['text']
    DocType = data['docType']

    pipe_nb = Pipeline(('vectorizer', BagOfWords(lowercase=True)), ('nb', MultinomialNB()))
    pipe_nb = pipe_nb.learn_one(txt, DocType)

    with open('mbn_classifier', 'wb') as picklefile:
        pickle.dump(pipe_nb, picklefile)

    res = {"status": "success"}
    return make_response(jsonify(res), 200)

# app route for update
@app.route('/update', methods = ['POST'])
def update():
    data = request.get_json();
    txt = data['text']


    DocType = data['docType']
    pipe_nb = pickle.load(open('mbn_classifier', 'rb'))
    df3 = pd.DataFrame([[txt, DocType]], columns=["Text", "DocType"])
    data_test = df3.to_records(index=False)
    for Text, DocType in data_test:
        y_pred_before = pipe_nb.predict_one(Text)
      #  metric = metrics.update(Text, y_pred_before)
        pipe_nb = pipe_nb.learn_one(Text, DocType)
        result_pred = pipe_nb.predict_one(Text)
        print(result_pred)


    with open('mbn_classifier', 'wb') as picklefile:
        pickle.dump(pipe_nb, picklefile)

    res = {"status": "success"}
    return make_response(jsonify(res), 200)



@app.route('/classify', methods = ['POST'])
def classify():
    data = request.get_json();
    txt = data['text']



    pipe_nb = pickle.load(open('mbn_classifier', 'rb'))
    result_pred = pipe_nb.predict_one(txt)

    res = {"result": result_pred}


    return make_response(jsonify(res), 200)





if __name__ == '__main__':
    app.debug = True
    app.run()


