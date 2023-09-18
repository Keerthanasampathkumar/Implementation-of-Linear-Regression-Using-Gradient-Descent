# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. import numpy as np.
3. Give the header to the data.
4. Find the profit of population.
5. Plot the required graph for both for Gradient Descent Graph and Prediction Graph.
6. End the program.

## Program:
```c
Program to implement the linear regression using gradient descent.
Developed by: KEERTHANA S
RegisterNumber:  212222230066

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("ex1.txt",header=None)
data

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Predication")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(x,y,theta)

def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=x.dot(theta)
    error=np.dot(x.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(x,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000s")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
### DATASET:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119477890/3286a82f-ed78-4311-ae84-821c264a4aab)
### Plt.profitprediction:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119477890/db84d110-47ab-4680-b8b7-8d1bde30a319)
### computeCost:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119477890/92b7bc5e-d055-4b2d-b90d-2005ab7dd502)
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119477890/a735ffae-8dbe-42ef-b8a3-2e3509f0a142)
### Cost function using Gradient Descent:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119477890/7ecb200a-856e-4b6d-90e8-005372f7bc46)
### Profit Prediction:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119477890/d9cb185c-9950-40f4-a2fb-807af1e42020)
### predict1:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119477890/cf41caed-d6c2-4220-ba14-a0e71b26a995)
### predict2:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119477890/91c4f9b8-1248-4e87-b5b8-23a442d3cea5)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
