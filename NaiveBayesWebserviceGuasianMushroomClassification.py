import numpy as np
import pandas as pd
import pickle
from flask import *
dataset = pd.read_csv('mushrooms.csv')
dataset
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in dataset.columns:
    dataset[col]=labelencoder.fit_transform(dataset[col])
Y = dataset['class']
X = dataset.drop(columns = 'class')
from sklearn.naive_bayes import GaussianNB
model_nb = GaussianNB()
model_nb.partial_fit(X, Y,classes=np.unique(Y))

with open('model_nb', 'wb') as picklefile:
    pickle.dump(model_nb,picklefile)


app = Flask(__name__)
# app route for update
@app.route('/update', methods = ['POST'])
def update():
    from sklearn.naive_bayes import GaussianNB
    data = request.get_json();
    data
    dataset1 = pd.read_csv('mushrooms.csv')
    df = pd.DataFrame(data, index=[0])
    df_update = pd.concat([dataset1,df], ignore_index=True)
    df_update.to_csv('mushroom_update.csv')
    model = GaussianNB()
    for col in df_update.columns:
        df_update[col] = labelencoder.fit_transform(df_update[col])
    Y_update = df_update['class']
    X_update = df_update.drop(columns='class')
    from sklearn.naive_bayes import GaussianNB
    model.partial_fit(X_update, Y_update, classes=np.unique(Y_update))
    import pickle
    with open('model_nb', 'wb') as picklefile:
        pickle.dump(model_nb, picklefile)

        res = {"status": "success"}
        return make_response(jsonify(res), 200)

@app.route('/classify', methods = ['POST'])
def classify():
    data = request.get_json();
    df_clsfy = pd.DataFrame(data,index = [0])
    le = LabelEncoder()
    for col in dataset.columns:
        dataset[col] = labelencoder.fit_transform(dataset[col])
    X_pred = df_clsfy
    model = pickle.load(('model_nb', 'rb'))
    result_pred =  model.predict(X_pred)
    res = {"result": result_pred}

if __name__ == '__main__':
    app.debug = True
    app.run()

