from flask import Flask,render_template,request
import pickle
import numpy as np

with open("model.pkl",'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/iris',methods=['POST'])
def predict():
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    bmi = float(request.form['bmi'])
    children = float(request.form['children'])
    smoker = float(request.form['smoker'])

    data = np.array([age,sex,bmi,children,smoker],ndmin=2)
    result = str(model.predict(data))
    
    return render_template('index.html',result=result[1:-2])


if __name__ == "__main__":
    app.run(host="0.0.0.0",port='8080',debug=False)