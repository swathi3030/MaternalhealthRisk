import numpy
import joblib

model=joblib.load('knncropmodel.pkl')

testdata=numpy.array([[25,130,80,15,98,86]])

result=model.predict(testdata)

print(f"Result of prediction={result[0]}")