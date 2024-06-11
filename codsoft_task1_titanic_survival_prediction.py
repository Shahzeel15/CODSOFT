##  CODSOFT ##
## TASK 1 : TITANIC SURVIVAL PREDICTION


import pandas as pd 
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
import streamlit as st
#read titanic.csv file
df=pd.read_csv("titanic.csv")
print(df.head())
# now we are dropping unnecessary columns from the data frame and taking only those columns which helps us to predict
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
print(df.head())

# assiging 0 to male and 1 to female
d={'male':0,'female':1}
df['Gender']=df['Gender'].map(d)
print(df.head())
# now adding survived column in the target var
target = df.Survived
print(target)
# droping the survived column and taking all other columns as inputs var means inputs having 
#pclass,age,genderand fare columns
 
inputs=df.drop('Survived',axis='columns')
print(inputs)

#inputs.columns[inputs.isna().any()]
# handling missing values if age column of inputs var having any missing values then it will be handled by it's mean
inputs.Age=inputs.Age.fillna(inputs.Age.mean())
print(inputs.head(10))

#now dividing it into traing and testing part where test size is 0.2 and storing in into
# x_train,x_test,y_train,y_test variables
# using train_test_split function which takes inputs var as independent var and target var as depended var and spit
x_train,x_test,y_train,y_test = train_test_split(inputs,target,test_size=0.2)
#load the model
model= GaussianNB()
# fit the model while passing x tarin and y train as argument
model.fit(x_train,y_train)
#printing it's score
print(model.score(x_test,y_test))
# predicting the top 20 values from y test to  check model's accuracy
print(y_test[:20])
print(model.predict(x_test)[:20])

# now create one streamlit application
st.header(":blue[Titanic Survival Prediction]")

# taking the value dynamically from the user to predict
in_class=(st.number_input(":violet[Enter Pclass : ]"))
gender=st.selectbox(":violet[Enter Gender : ]",options=['male','female'])
if(gender=='male'):
    in_gender=0
else:
    in_gender=1
#st.write(in_gender)
in_age=(st.number_input(":violet[Enter Age : ]"))
in_fare=(st.number_input(":violet[Enter Fare : ]"))
in_age=float(in_age)
in_fare=float(in_fare)

# adding those values which we took from the user in test dataframe
test = pd.DataFrame({'Pclass': [in_class],'Gender':[in_gender], 'Age': [in_age],'Fare':[in_fare]})
# test = pd.DataFrame({'Pclass': [1],'Gender':[1], 'Age': [82.000000],'Fare':[23.2750]})
# and based on that values it will predict the outcome which will print on streamlit that where it survived or not
a=model.predict(test)

#if outcomne is 0 then it's not survived else it survived based on the user data
if st.button("Prediction"):         
         if(a==0):
             st.subheader(":red[NOT SURVIVED]")
             print("not survived")
         
         else:
             st.subheader(":green[SURVIVED]")
             print("survived")
