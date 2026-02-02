from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        size = float(request.form["size"])
        bedrooms = float(request.form["bedrooms"])
        age = float(request.form["age"])

        X = np.array([[size, bedrooms, age]])
        prediction = model.predict(X)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)