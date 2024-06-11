##  CODSOFT ##
## TASK 3 : IRIS flower classification ##


import numpy as np  # to deal with arrays  
import matplotlib.pyplot as plt # ploting graphs   
import seaborn as sns # data visualization tool   
import pandas as pd # data frames & data manupulation 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
import streamlit as st  


# load the dataset 
df_iris=pd.read_csv('IRIS.csv')  
print(df_iris.head()) # printing head of the dataset 
print(df_iris.shape) # shape of the dataframe total 151 rows & 5 columns are there
print(df_iris.describe()) #  checks statics & analysis for the dataframe 

df_iris=df_iris.values

# taking first four columns as input var which are indepented attributes
inputs=df_iris[:,0:4]
# and last column species as op which is depended var
op=df_iris[:,4]

# passing the inputs and op to split the data into train and test where test size is 0.2
x_train,x_test,y_train,y_test=train_test_split(inputs,op,test_size=0.2)

# load the svc model for support vector classification
model = SVC()
# fit the model
model.fit(x_train,y_train)
# streamlit webapplication 
st.title("IRIS Flower Classification")
st.info("Enter the value of sepal_length sepal_width petal_length petal_width as input to predict the species of the flower")

# taking user input for classification and prediction
sepal_length=st.number_input("Enter sepal length : ")
sepal_width=st.number_input("Enter sepal width : ")
petal_length=st.number_input("Enter petal length : ")
petal_width=st.number_input("Enter petal width : ")
# converting them into float
sepal_length=float(sepal_length)
sepal_width=float(sepal_width)
petal_length=float(petal_length)
petal_width=float(petal_width)
# puttng it into one test dataframe to predict the outcome
test=pd.DataFrame({'sepal_length':[sepal_length],'sepal_width':[sepal_width],'petal_length':[petal_length],'petal_width':[petal_width]})
# now use the model that we fit before and pass the test which is user input's value and based on that predict the outcome

prediction=model.predict(test)

# now when user click on these btn it will show the flower classes's prediction
prediction_btn=st.button("Flower species prediction")
if prediction_btn:
    st.subheader(prediction)

# on click of these btn it will display the accuracy score
accuracy_btn=st.button("Accuracy score")
if accuracy_btn:
    # prediction based on the xtest data
    y_predict=model.predict(x_test)
    # pass the actual ytest data + ypredicted data to check the accuracy of our model
    from sklearn.metrics import accuracy_score
    st.write("Accuracy Score Of The Model : ")
    st.write(accuracy_score(y_test,y_predict)*100)

# on click of these button it will display classification report
class_report=st.button("Check Flower Classification Report ")
if class_report:
    # prediction based on the xtest data
    y_predict=model.predict(x_test)
    from sklearn.metrics import classification_report
    st.subheader("The Classification Report Of The Model : ")
     # pass the actual ytest data + ypredicted data to check the classification report
    st.write(classification_report(y_test,y_predict))


