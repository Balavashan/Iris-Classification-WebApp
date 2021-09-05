from flask import Flask,render_template,request
import pickle as pkl
import numpy as np

app=Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")




@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        x=float(request.form.get('petallength'))
        y=float(request.form.get("sepallength"))
        w=float(request.form.get("petalwidth"))
        z=float(request.form.get("sepalwidth"))

    
    with open('model.pkl','rb') as f:
        model=pkl.load(f)
    arr=np.array([x,y,w,z]).reshape(1,-1)
    predict=model.predict(arr)
    flower=["Iris-setosa","Iris-versicolor","","Iris-virginica"]

    return render_template('index.html',data=flower[predict[0]])



if __name__=='__main__':
    app.run(debug=True)