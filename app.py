
import flask
from flask import Flask, request, render_template
import pickle
import joblib
import pandas as pd

app = Flask(__name__)
housepriceprediction = None
# Load the trained model
model = pickle.load(open('housepriceprediction.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input values from the form
        area = flask.request.form['area']
        bhk  = flask.request.form['bhk']
        year = flask.request.form['year']
        type = flask.request.form['type']
        # return str([area,bhk,year,type])
        new_house = pd.DataFrame({'squarefeet area':[int(area)],'BHK':[int(bhk)],'year built':[int(year)]})
        model=joblib.load("asma.model_ml")
        prediction=model.predict(new_house)
        return render_template("index.html",price=round(prediction[0],1))
      

    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
