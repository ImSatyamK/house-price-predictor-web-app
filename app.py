from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        crime_rate = float(request.form["crime_rate"])
        residential_land = float(request.form["residential_land"])
        industrial_area = float(request.form["industrial_area"])
        near_river = float(request.form["near_river"])
        air_pollution = float(request.form["air_pollution"])
        avg_rooms = float(request.form["avg_rooms"])
        old_houses = float(request.form["old_houses"])
        distance_to_city = float(request.form["distance_to_city"])
        highway_access = float(request.form["highway_access"])
        property_tax = float(request.form["property_tax"])
        pupil_teacher_ratio = float(request.form["pupil_teacher_ratio"])
        black_population_index = float(request.form["black_population_index"])
        lower_income_pct = float(request.form["lower_income_pct"])

        X = np.array([[
            crime_rate,
            residential_land,
            industrial_area,
            near_river,
            air_pollution,
            avg_rooms,
            old_houses,
            distance_to_city,
            highway_access,
            property_tax,
            pupil_teacher_ratio,
            black_population_index,
            lower_income_pct
        ]])

        prediction = model.predict(X)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)