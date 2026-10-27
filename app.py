
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load models exported from Google Colab
saved = joblib.load("insurance_models.pkl")
classifier = saved["classifier"]
regressor = saved["regressor"]
threshold = saved["threshold"]
features = saved["features"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        sex = request.form["sex"].lower()
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = request.form["smoker"].lower()
        region = request.form["region"].lower()

        # Validate input values
        if not 18 <= age <= 100:
            raise ValueError("Age must be between 18 and 100.")

        if sex not in ["male", "female"]:
            raise ValueError("Select a valid gender.")

        if not 10 <= bmi <= 80:
            raise ValueError("BMI must be between 10 and 80.")

        if not 0 <= children <= 20:
            raise ValueError("Children must be between 0 and 20.")

        if smoker not in ["yes", "no"]:
            raise ValueError("Select a valid smoking status.")

        if region not in [
            "southeast", "southwest", "northwest", "northeast"
        ]:
            raise ValueError("Select a valid region.")

        # Arrange features in the same format used during training
        person = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }], columns=features)

        # Random Forest predictions
        category = int(classifier.predict(person)[0])
        charges = float(regressor.predict(person)[0])

        prediction = (
            "High Cost" if category == 1 else "Low Cost"
        )

        return render_template(
            "index.html",
            prediction=prediction,
            grade=max(0, charges)
        )

    except (ValueError, KeyError) as e:
        return render_template(
            "index.html",
            error=str(e) or "Please enter valid details."
        )


if __name__ == "__main__":
    app.run(debug=True)
