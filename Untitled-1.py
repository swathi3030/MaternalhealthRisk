# %%
import pandas as pd

# %%
dataset=pd.read_csv('Maternal.csv')
dataset

# %%
x=dataset.iloc[:,[0,1,2,3,4,5]].values

# %%
x[0]

# %%
y=dataset.iloc[:,[6]].values

# %%
y[0]

# %%
from sklearn.model_selection import train_test_split

# %%
model=KNeighborsClassifier()

# %%
model=KNeighborsClassifier()

# %%
import numpy as np
ytrain=np.ravel(ytrain)
model.fit(xtrain,ytrain)

# %%
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=100)

# %%
print(xtrain[0],len(xtrain))

# %%
print(xtest[0],len(xtest))

# %%
from sklearn.neighbors import KNeighborsClassifier

# %%
model=KNeighborsClassifier()

# %%
import numpy as np
ytrain=np.ravel(ytrain)
model.fit(xtrain,ytrain)

# %%
testdata=np.array([[25,130,80,15,98,86]])

# %%
result=model.predict(testdata)
result

# %%
ypred=model.predict(xtest)

# %%
print(ypred)

# %%
from sklearn.metrics import accuracy_score

# %%
score=accuracy_score(ytest,ypred)
score*100

# %%
import joblib
joblib.dump(model,"knncropmodel.pkl")

# %%


