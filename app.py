from doctest import debug
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("model", "rb"))

@app.route("/")
def home():
    return render_template("index.html",prediction_text = "predict")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)[0]
    if prediction == 1:
        return render_template("index.html", prediction_text = "Yes")
    elif prediction == 0:
        return render_template("index.html", prediction_text = "No")
    else:
        return render_template("index.html", prediction_text = "Error")

if __name__ == '__main__':
    app.run(debug=True)









# return render_template("index.html", prediction_text = "The prediction is {}".format(prediction))