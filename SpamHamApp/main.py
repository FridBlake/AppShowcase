import pickle
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, request, redirect, url_for, flash


app = Flask(__name__)

with open("countvec.pkl", "rb") as f:
    countvec = pickle.load(f)

with open("nb_model.pkl", "rb") as g:
    classifier = pickle.load(g)

def predict(val):
    predictiondata = countvec.transform([val]).toarray()
    return("Ham" if classifier.predict(predictiondata) == 0 else "Spam")



@app.route("/")
def home():
    return render_template("home.html", title = "Spam-Ham detection!")

@app.route("/result", methods=['POST', 'GET'])
def result():
    if request.method == "POST":
        text = request.form['entry']
        return render_template("result.html", text=predict(text))
    else:
        return "<p> ERROR :: DID YOU TRY TO USE A GET REQUEST? EXCUSE ME!? </p>"

app.run(debug=True, port=5000)