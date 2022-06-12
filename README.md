# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Senthil Kumar S
RegisterNumber:  212221230091
*/
import pandas as pd
df=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semster 2/Intro to ML/Placement_Data.csv")
df.head()
df.tail()
df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()
df1.isnull().sum()
#to check any empty values are there
df1.duplicated().sum()
#to check if there are any repeted values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1["gender"] = le.fit_transform(df1["gender"])
df1["ssc_b"] = le.fit_transform(df1["ssc_b"])
df1["hsc_b"] = le.fit_transform(df1["hsc_b"])
df1["hsc_s"] = le.fit_transform(df1["hsc_s"])
df1["degree_t"] = le.fit_transform(df1["degree_t"])
df1["workex"] = le.fit_transform(df1["workex"])
df1["specialisation"] = le.fit_transform(df1["specialisation"])
df1["status"] = le.fit_transform(df1["status"])
df1
x=df1.iloc[:,:-1]
x
y = df1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.09,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
#liblinear is library for large linear classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))
```

## Output:

## Original data(first five columns):

![Capture](https://user-images.githubusercontent.com/93860256/173221664-cb3d48a5-c07a-4d03-aa51-9421a7da3041.PNG)

## Data after dropping unwanted columns(first five):

![2](https://user-images.githubusercontent.com/93860256/173221702-83e0327f-0552-497c-bab5-39657c2fe8d0.PNG)


## Checking the presence of null values:

![3](https://user-images.githubusercontent.com/93860256/173221716-3678cac7-2f14-4e35-985e-e14de701e16e.PNG)


## Checking the presence of duplicated values:

![4](https://user-images.githubusercontent.com/93860256/173221731-80e4a234-22d9-4c67-93c4-d735ec3779ef.PNG)


## Data after Encoding:

![5](https://user-images.githubusercontent.com/93860256/173221743-701057ce-274d-4094-afec-71e2df2d64e8.PNG)


## X Data:

![6](https://user-images.githubusercontent.com/93860256/173221750-eb884c2f-d009-4a62-92f5-952735668019.PNG)


## Y Data:

![7](https://user-images.githubusercontent.com/93860256/173221761-141e4a28-2353-4c55-adc5-081d7dd495c6.PNG)


## Predicted Values:

![8](https://user-images.githubusercontent.com/93860256/173221778-6bea10a7-7bc0-4786-aa1a-b6e687c52e3a.PNG)


## Accuracy Score:

![9](https://user-images.githubusercontent.com/93860256/173221796-ffc426dd-a2ad-4ef3-8eca-42aa8eb26279.PNG)


## Confusion Matrix:

![10](https://user-images.githubusercontent.com/93860256/173221813-73192ca6-3eee-4ead-ae84-922b524e5399.PNG)


## Classification Report:

![11](https://user-images.githubusercontent.com/93860256/173221824-826b017d-3829-45e5-844b-299158681dd0.PNG)


## Predicting output from Regression Model:

![12](https://user-images.githubusercontent.com/93860256/173221834-4ff796c3-82a5-474c-927b-83589e4d4270.PNG)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
