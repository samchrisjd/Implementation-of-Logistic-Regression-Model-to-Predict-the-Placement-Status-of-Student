# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import Dataset
Load the dataset using pandas and create a copy for processing.
2. Drop Irrelevant Columns
Remove columns like sl_no and salary as they are not needed for prediction.
3. Check for Missing and Duplicate Data
Use .isnull().sum() and .duplicated().sum() to identify and handle missing or duplicate entries.
4. Encode Categorical Features
Convert all categorical columns (e.g., gender, board types, specialisation, etc.) into numeric form using LabelEncoder.
5. Split Features and Target Variable
Set x as input features (all columns except status) and y as the target (status).
6. Split Dataset into Train and Test Sets
Use train_test_split() to divide the data into training and testing sets (e.g., 80% train, 20% test).
7. Train the Logistic Regression Model
Initialize and train the logistic regression model using x_train and y_train.
8. Predict and Evaluate
Predict outcomes on x_test and evaluate performance using accuracy, confusion matrix, and classification report.
9. Make a New Prediction
Use .predict() with a new input list of values to predict placement status for a new student.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SAM CHRIS M
RegisterNumber:  212224040285
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])

x=data1.iloc[:,:-1]
x
print(x)
y=data1["status"]
y
print(y)
print()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
print()
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
print()
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
print()
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

<img width="882" height="621" alt="image" src="https://github.com/user-attachments/assets/3258c426-e56e-47d6-abf9-03af18470122" />

<img width="1165" height="665" alt="image" src="https://github.com/user-attachments/assets/0071a4b2-d5d6-49f7-88d2-28563f69440e" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
