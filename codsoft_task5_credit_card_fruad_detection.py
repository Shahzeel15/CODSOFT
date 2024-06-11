##  CODSOFT ##
## Task 5 : credit card fruad detection



# logistic regression 
import numpy as np 
import pandas as pd # data frame & data cleaning 
from sklearn.model_selection import train_test_split # spliting the data in training & testing part
from sklearn.linear_model import LogisticRegression # for logistic regression model
from sklearn.metrics import accuracy_score  # to check the accuracy of the model 
import streamlit as st
import matplotlib.pyplot as plt

# read the data from the csv and storing is into dataframe
df_credicard=pd.read_csv('creditcard.csv')
print(df_credicard.head())# printing the head of the dataframe
print(df_credicard.info()) # printing info and we can see all features are of float type
print(df_credicard.shape)# printing the shape of df to check that howmany rows and columns are there here it's (284807, 31)
# these function prints the value count of the column class to verify whether the data is balanced or not our data 
# is almost imbalance because it has    284315 o's and 492 1's
print(df_credicard['Class'].value_counts()) 
# let's check statics for the data
print(df_credicard.describe())

# now As our dataset is imbalanced we need to make it balanced
#creating two separate datasets 
# first for legit transacation where the value of class is 0
legit_trans = df_credicard[df_credicard['Class']==0]
# And second is for fruad transaction where the value of class is 1
fraud_trans = df_credicard[df_credicard['Class']==1]
#putting those two into seperate csv so that it helps to verify output
# legit_trans.to_csv('legit.csv') 
# fraud_trans.to_csv('fruad.csv')
print(legit_trans.shape)#(284315, 31)
print(fraud_trans.shape)# (492, 31)

# taking same 492 values as sample for legit too to making it balanced
legit_sample=legit_trans.sample(n=492)
df_credicard=pd.concat([legit_sample,fraud_trans],axis=0) # adding those two into df
print(df_credicard.shape) # now we do have total 984 data of both 492 of legit and 492 of fraud
print(df_credicard['Class'].value_counts())

# calculating mean of class for 0 value and 1 value
df_credicard.groupby('Class').mean()

#split the data
x = df_credicard.drop('Class',axis=1) # indepedented var x having all the list of features other then class cloumn
y=df_credicard['Class'] # y is depended var it has value of class column

print(x.shape,y.shape) # (984, 30) (984,)
# now split it into training and testing part 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

# load and fit the logistic regression model
model = LogisticRegression()
model.fit(x_train,y_train)
# predict the value for reaming testing value
ypred= model.predict(x_test)
# fr comparasion btw ypread and ytest we use accuracy score
print(accuracy_score(ypred,y_test)) # for our model it's 0.9035532994923858 accuracy
print(accuracy_score(model.predict(x_train),y_train) )# for training accuracy score
     
# streamlit webapplication 
st.title("Credit Card Fraud Detection Model")
input_data=st.text_input("Enter All the values of Required Features : ")
input_data_split=input_data.split(',')

prediction = st.button("check prediction")
if prediction:
    features_df=np.asarray(input_data_split,dtype=np.float64)
    prediction=model.predict(features_df.reshape(1,-1))
    if prediction[0]==0:
        st.write(":green[Legitimate Transaction]")
    else :
        st.write(":red[Fradulent Transaction]")


