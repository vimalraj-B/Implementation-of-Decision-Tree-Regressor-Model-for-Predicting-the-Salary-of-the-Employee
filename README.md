# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Load and preprocess the employee salary dataset (handle missing values, encode categorical data).
2. Split the dataset into features (X) and target (y), then into training and testing sets.
3. Initialize the DecisionTreeRegressor model with appropriate hyperparameters.
4. Train the model using the training dataset. 5.Evaluate the model on the test dataset and predict employee salaries.
```

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: VIMALRAJ B
RegisterNumber:  212224230304
*/

import pandas as pd
 data=pd.read_csv(r"C:\Users\admin\Downloads\Salary.csv")
 print(data.head())
 data.info()
 data.isnull().sum()    
 from sklearn.preprocessing import LabelEncoder
 le=LabelEncoder()
 data["Position"]=le.fit_transform(data["Position"])
 print(data.head())
 x=data[["Position","Level"]]
 print(x.head())
 y=data["Salary"]
 print(y.head())
 from sklearn.preprocessing import LabelEncoder
 le=LabelEncoder()
 data["Position"]=le.fit_transform(data["Position"])
 print(data.head())
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
 from sklearn.tree import DecisionTreeRegressor
 dt=DecisionTreeRegressor()
 dt.fit(x_train,y_train)
 y_pred=dt.predict(x_test)
 print(y_pred)
 from sklearn import metrics
 mse=metrics.mean_squared_error(y_test,y_pred)
 print(mse)
 r2=metrics.r2_score(y_test,y_pred)
 print(r2)
 dt.predict([[5,6]])

```

## Output:

![Screenshot 2025-05-14 110657](https://github.com/user-attachments/assets/6d81328a-1c98-40b8-90f1-311d850f8caa)

![Screenshot 2025-05-14 110703](https://github.com/user-attachments/assets/a3839d21-a491-404e-bb0a-081b9d852830)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
