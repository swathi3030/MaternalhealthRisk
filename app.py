from flask import Flask, render_template,request 
import numpy as np
import joblib
app=Flask(__name__)

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        a=float(request.form['a'])
        bp=float(request.form['bp'])
        dbp=float(request.form['dbp'])
        bs=float(request.form['bs'])
        bt=float(request.form['bt'])
        hr=float(request.form['hr'])
        
        
        
        testdata=np.array([[a,bp,dbp,bs,bt,hr]])
        model=joblib.load("knncropmodel.pkl")
        res=model.predict(testdata)
        cropname=res[0]
        print(f"predicted result = {cropname}")
        
        return render_template('index.html',result=cropname)
@app.route('/')
def index():
    return render_template('index.html')

if __name__=='__main__':
        app.run(debug=True)
