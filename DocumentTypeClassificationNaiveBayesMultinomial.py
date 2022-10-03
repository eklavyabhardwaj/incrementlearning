import pickle
import pandas as pd
import river
from river.naive_bayes import MultinomialNB
from river.feature_extraction import BagOfWords, TFIDF
from river.compose import Pipeline
from flask import Flask, render_template, request, Response, jsonify, make_response
import river
from river.naive_bayes import MultinomialNB
from river.feature_extraction import BagOfWords, TFIDF
from river.compose import Pipeline
import pandas as pd
from sklearn.datasets import load_files
# Build Pipeline
from river import metrics
app = Flask(__name__)


phoenix_dataset = load_files(r"C:/Users/eklav/RiverPython Project/DocumentTypeClassification/DocumentTypes")
data, target = phoenix_dataset.data, phoenix_dataset.target

X = data
y = target
global Text
global DocType

dataset = {'Text': X, 'DocType': y}


import pandas as pd
df = pd.DataFrame(dataset)

df.to_csv('df_output.csv', encoding='utf-8')
df1 = pd.read_csv('df_output.csv')
df1 = df1.drop(['Unnamed: 0'], axis =1)



# Convert to Tuple
data = df1.to_records(index=False)

pipe_nb = Pipeline(('vectorizer',BagOfWords(lowercase=True)),('nb',MultinomialNB()))





# Train
for Text,DocType in data:
    pipe_nb = pipe_nb.learn_one(Text,DocType)


with open('mbn_classifier', 'wb') as picklefile:
    pickle.dump(pipe_nb, picklefile)

app = Flask(__name__)


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
    df4 = pd.DataFrame(txt)
    df4.to_csv('df_input.csv', encoding='utf-8')
    df5 = pd.read_csv('df_input.csv')
    df5 = df5.drop(['Unnamed: 0'], axis=1)
    data_test = df5.to_records(index=False)
    for Text in data_test:
        result_pred = pipe_nb.predict_one(Text)
        print(result_pred)

    result_pred = pipe_nb.predict_one(txt)
    res = {"result": result_pred}



    return make_response(jsonify(res), 200)





if __name__ == '__main__':
    app.debug = True
    app.run()
