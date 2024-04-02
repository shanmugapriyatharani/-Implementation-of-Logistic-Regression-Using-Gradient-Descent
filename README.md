# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages.
2.Read the given dataset and assign x and y array.
3.Split x and y into training and test set. 
4.Scale the x variables.
5.Fit the logistic regression for the training set to predict y.
6.Create the confusion matrix and find the accuracy score, recall sensitivity and specificity.
7.Plot the training set results. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SHANMUGA PRIYA.T
RegisterNumber: 212222040153
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#reading and displaying dataframe
df=pd.read_csv("Social_Network_Ads (1).csv")
df
x=df.iloc[:,[2,3]].values
y=df.iloc[:,4].values 
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.fit_transform(xtest)
from sklearn.linear_model import LogisticRegression
c=LogisticRegression(random_state=0)
c.fit(xtrain,ytrain)
ypred=c.predict(xtest)
ypred
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
cm
from sklearn import metrics
acc=metrics.accuracy_score(ytest,ypred)
acc
r_sens=metrics.recall_score(ytest,ypred,pos_label=1)
r_spec=metrics.recall_score(ytest,ypred,pos_label=0)
r_sens,r_spec
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
xs,ys=xtrain,ytrain
x1,x2=np.meshgrid(np.arange(start=xs[:,0].min()-1,stop=xs[:,0].max()+1,step=0.01),
               np.arange(start=xs[:,1].min()-1,stop=xs[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,c.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                            alpha=0.75,cmap=ListedColormap(('skyblue','green')))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x1.max())
for i,j in enumerate(np.unique(ys)):
    plt.scatter(xs[ys==j,0],xs[ys==j,1],
                c=ListedColormap(('black','white'))(i),label=j)
plt.title("Logistic Regression(Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

```

## Output:
## Array of X:

![image](https://github.com/shanmugapriyatharani/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393427/4b81e5aa-fd0a-4311-8fc9-5269eb1ff783)

## Array of Y:

![image](https://github.com/shanmugapriyatharani/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393427/bf940b96-4a19-47a8-8fdc-0769fd473a20)

## Score Graph:

![image](https://github.com/shanmugapriyatharani/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393427/fc6a783d-0bcb-4c2b-8fd9-a41ba58895ff)

## Sigmoid Function Graph:

![image](https://github.com/shanmugapriyatharani/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393427/da34418d-f238-43ed-8ac6-ce5ab9fe1eb1)

## X_train_grad Value:

![image](https://github.com/shanmugapriyatharani/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393427/9fbce397-e76f-4e94-b944-0223bc3fd792)

## Y_train_grad Value:

![image](https://github.com/shanmugapriyatharani/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393427/e7767e02-dde9-48f8-b199-1449fa159fe2)

## Print res_X:

![image](https://github.com/shanmugapriyatharani/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393427/0c404ff5-72d8-47c6-b93b-628f3e13b60b)

## Decision boundary:

![image](https://github.com/shanmugapriyatharani/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393427/8e772817-fd6e-473f-a1d2-83cac17baf25)

## Probability Value:

![image](https://github.com/shanmugapriyatharani/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393427/39a12a8a-4870-496c-a7cd-138060011527)

Prediction Value of Mean:
![image](https://github.com/shanmugapriyatharani/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393427/c47a6bc9-032e-4e1c-9df9-5ab96c956dbe)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

