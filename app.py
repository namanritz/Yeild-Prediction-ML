from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn

print(sklearn.__version__)
# loading models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Crop_Year = request.form['Crop_Year']
        Annual_Rainfall = request.form['Annual_Rainfall']
        Pesticide = request.form['Pesticide']
        Fertilizer = request.form['Fertilizer']
        State = request.form['State']
        Area = request.form['Area']
        Crop = request.form['Crop']

        features = np.array([[Crop_Year, Annual_Rainfall, Pesticide, Fertilizer, State, Area, Crop]])
        transformed_features = preprocessor.transform(features)
        predicted_yield = dtr.predict(transformed_features).reshape(1, -1)

        return render_template('index.html', prediction=predicted_yield)


if __name__ == "__main__":
    app.run(debug=True)
