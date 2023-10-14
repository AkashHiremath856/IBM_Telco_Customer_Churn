from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Load the pickled model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get the input values from the form
    gender = request.form["gender"]
    age = int(request.form["age"])
    dependents = int(request.form["dependents"])
    married = request.form["married"]
    monthly_charge = float(request.form["monthly_charge"])
    city = request.form["city"]
    extra_data_charges = float(request.form["extra_data_charges"])
    unlimited_data = request.form["unlimited_data"]
    satisfaction_score = int(request.form["satisfaction_score"])
    churn_score = int(request.form["churn_score"])

    # Create a DataFrame with the input data
    data = pd.DataFrame(
        {
            "Gender": [gender],
            "Age": [age],
            "Number of Dependents": [dependents],
            "Married": [married],
            "Monthly Charge": [monthly_charge],
            "City": [city],
            "Total Extra Data Charges": [extra_data_charges],
            "Unlimited Data": [unlimited_data],
            "Satisfaction Score": [satisfaction_score],
            "Churn Score": [churn_score],
        }
    )

    encoder = LabelEncoder()

    data.loc[:, "Gender"] = encoder.fit_transform(data.Gender)
    data.loc[:, "Unlimited Data"] = encoder.fit_transform(data["Unlimited Data"])
    data.loc[:, "Married"] = encoder.fit_transform(data.Married)
    data.loc[:, "City"] = encoder.fit_transform(data["City"])

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    prediction = model.predict(data)

    return render_template(
        "index.html", prediction="Churn" if prediction[0] == 1 else "Not Churn"
    )


if __name__ == "__main__":
    app.run(debug=True)
