###  CODSOFT ##
#Task 4 : Sales Prediction
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Sales Prediction")
st.title('Sales Prediction')
# read dataframe
df_sales=pd.read_csv("advertising.csv")
print(df_sales.head())


tab1, tab2 , tab3 , tab4, tab5= st.tabs(['Data Analysics','Pairpost Visualization','Histogram Visualization','Heatmap Visualization','Prediction'])

with tab1:      
      st.subheader('Data Analysics')

      t1, t2= st.tabs(['Data Frame','Data describe'])
      with t1:
            st.subheader("Data Frame ")
            
            st.write(df_sales)
           
             # shape of the  dataframe of advertising is (200, 4) means it has 4 rows and 200 columns
      with t2:     
           st.subheader("Data Describe")
           st.info("It describes the statical performance of the Data")
           st.write("Shape of the Data Frame is :  ", df_sales.shape)
           
           print(df_sales.describe()) # it describes the dataframe with it's statical performance
           st.write(df_sales.describe())
           st.info("Observation : From the above describe function we observed that Avg expense spend is highest on Tv and lowest on Radio where maximum sale is 27 and minimum sale is 1.6")



with tab2:
     
#visualiation 
# pairplot

        
    # sns.pairplot(df_sales,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',kind='scatter')
    # plt.show()
  
       st.subheader("Pair Plot of Sales vs. Marketing Channels")
       fig = sns.pairplot(df_sales, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', kind='scatter')
       st.pyplot(fig)
       st.info("Pair Plot Observation : we observed  from above pair plot is when advertising cost increases in TV Ads the sales will increase as well where as the for newspaer and Radio it bit unpredictable")



# so by these pairplot we can identify that when the advertising increase tv sales increase too where as 
#the radio & newspaper is sactterd means bit unpredictable

# histogram of tv
with tab3:  
        st.subheader("Histogram For visualization")      
        t1, t2 , t3 = st.tabs(['hist1','hist2','hist3'])
        with t1:
             st.subheader("Histograms Charts For TV")
             plt.figure(figsize=(8,3))
             df_sales['TV'].plot.hist(bins=10,color="green")
             plt.xlabel("TV")
             plt.ylabel("Sales")
             plt.title("Sales based on Tv Advertising cost")
             st.pyplot(plt)
             plt.close()
        
        with t2:
             st.subheader("Histograms Charts For Radio")
             plt.figure(figsize=(8,3))
             df_sales['Radio'].plot.hist(bins=10,color="blue")
             plt.xlabel("Radio")
             plt.ylabel("Sales")
             plt.title("Sales based on Radio Advertising cost")
             st.pyplot(plt)
             plt.close()
        
        with t3:
             st.subheader("Histograms Charts For Newspaper")
             plt.figure(figsize=(8,3))
             df_sales['Newspaper'].plot.hist(bins=10,color="red")
             plt.xlabel("Newspaper")
             plt.ylabel("Sales")
             plt.title("Sales based on Newspaper Advertising cost")
             st.pyplot(plt)
             plt.close()
        st.info("Histogram Observation : The Majority is the result of low advertising cost in newspaper")

with tab4:     
#heatmap
     st.subheader("Heatmap For visualization")  
     corr_matrix = df_sales.corr()
     fig, ax = plt.subplots(figsize=(8, 3)) 
     sns.heatmap(corr_matrix, annot=True, ax=ax)
     st.pyplot(fig)
     st.info("Final observation : We can clearly see that Sales is highly coorelared with the Tv ")

# so by above visualization you can say that sale is highly corealted with tv

# scale and split the dataframe
from sklearn.model_selection import train_test_split
# spliting the dataset into training and testing where test size is 30 % and training part it is of 70%
x_train,x_test,y_train,y_test=train_test_split(df_sales[['TV']],df_sales[['Sales']],test_size=0.3,random_state=0)
print(x_train,y_train)
print(x_test,y_test)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

y_predict=model.predict(x_test)
print(y_predict)

print(model.coef_)
print(model.intercept_)
#plt.plot(y_predict)
with tab5:
      t1, t2= st.tabs(['Predicted Values','Make Prediction'])      
      with t1:
           # print the graphn of predicted value
           st.subheader("Predicted Values")
           plt.figure(figsize=(10,4))
           plt.plot(y_predict)
           plt.xlabel("Index") 
           plt.ylabel("Predicted Value")

           st.pyplot(plt)
      with t2:  
          # take the value from the user to predict the value  
          input_data=st.number_input("Enter All the values of Required Features : ")
          test = pd.DataFrame({'TV': [input_data]})
          a=model.predict(test)
          st.subheader(a)





