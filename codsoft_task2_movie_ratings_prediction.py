##  CODSOFT ##
## TASK 2 : MOVIE RATINGS PREDICTION 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import streamlit as st

st.title("MovieRatings Prediction")
# read dataframe
movie_ratings_df=pd.read_csv('task4.csv',encoding='unicode_escape') 
print(movie_ratings_df.head(10)) #print the top ten values 
print(movie_ratings_df.shape) # shape of the dataframe which is (15509, 10) means 11509 rows and 10 columns

tab1,tab2,tab3,tab4=st.tabs(['Data Analaysis','Data Visualization','Prediction','Performance metrix'])

with tab1:
        st.subheader("Data Analysis")
        t1,t2,t3=st.tabs(['Data Frame','Data Cleaning','Data preprocessing'])
        with t1:
                st.subheader("Data Frame ")
                st.write(movie_ratings_df)
                st.info("Shape of the Data frame Before Cleaning and Preprocessing : ");st.write(movie_ratings_df.shape)
        with t2:
                
                ## data cleaning ##        
                print(movie_ratings_df.isnull().sum()) # identify the total null values of each columns
                print(movie_ratings_df.info()) # getting the info set of the dataframe
                print(movie_ratings_df.duplicated().sum()) # to check the duplicated values
                # drop the nullvalues
                movie_ratings_df.dropna(inplace=True)
                print(movie_ratings_df.head(10))

                print(movie_ratings_df.shape) # now after dropping the nullvalues total remainig rows are 5659
                print(movie_ratings_df.isnull().sum()) # after drpping all the null values now it's get zero nullvalues in our dataframe

                # dropping the duplicated values too 
                movie_ratings_df.drop_duplicates(inplace=True)
                print(movie_ratings_df.head(10))
                print(movie_ratings_df.shape) # after dropping the duplicated value the shape is 5659

                print(movie_ratings_df.columns) # name of columns in our dataset
                st.subheader("Data Cleaning")
                st.info("After Cleaning the Data the shape of the Data frame is : ");st.write(movie_ratings_df.shape)
                st.write(movie_ratings_df)
        with t3:
                
               ## Data preprocessing ##
               # based on column wised requrirements 

               # eliminating  brackets from the column and kepping the simple values by replacing () with ''
               movie_ratings_df['Year']=movie_ratings_df['Year'].str.replace(r'[()]','',regex=True).astype(int)

               #eliminating min from the column duration by replacing min with blankspace
               movie_ratings_df['Duration']=pd.to_numeric(movie_ratings_df['Duration'].str.replace('min',''))

               # spliting the genre by "," to keep unique genres and replacing the  null values with mode
               movie_ratings_df['Genre']=movie_ratings_df['Genre'].str.split(',')
               movie_ratings_df=movie_ratings_df.explode('Genre')
               movie_ratings_df['Genre'].fillna(movie_ratings_df['Genre'].mode()[0],inplace=True)

               # replacing ',' with blankspace to remove , from the votes column
               movie_ratings_df['Votes']=pd.to_numeric(movie_ratings_df['Votes'].str.replace(',',''))

               #check info of dataframe
               print(movie_ratings_df.info())
               st.subheader("Data Preprocessing")
               st.info("Preprocesses the data by columns depending on it's requirements")
               st.info("After Preprocessing the Data the shape of the Data frame is : ");st.write(movie_ratings_df.shape)
               st.write(movie_ratings_df)
with tab2:
      st.subheader("Data Visualization")
      t1,t2,t3=st.tabs(['Histogram','Line Chart','Rating Distribution'])
      ## Data visualization ##
      with t1:       
             #histogram for the year column 
             year_hist=px.histogram(movie_ratings_df,x="Year",histnorm="probability density",nbins=30)
             year_hist.update_traces(marker_color='lightblue')
             year_hist.update_layout(title="Histogram of year",width=800,height=400)
             st.plotly_chart(year_hist)
             # it has increase the density with the period of years 

      with t2:
             avg_rating_by_year=movie_ratings_df.groupby(['Year','Genre'])['Rating'].mean().reset_index()
             top_genres=movie_ratings_df['Genre'].value_counts().head(10).index
             avg_ratings_by_year=avg_rating_by_year[avg_rating_by_year['Genre'].isin(top_genres)]
             fig=px.line(avg_rating_by_year,x='Year',y='Rating',color='Genre')
             fig.update_layout(title="AVG Ratings By Year For Top Genre",xaxis_title="Year",yaxis_title="AVG Ratings")
             st.plotly_chart(fig)
      # line chart 
      with t3:
            
             ratings=px.histogram(movie_ratings_df,x='Rating',histnorm='probability density',nbins=40)
             ratings.update_traces(marker_color='pink')
             ratings.update_layout(title="Rating Distribution",title_x=0.5,width=800,height=400)
             st.plotly_chart(ratings)

# prediction 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error,r2_score

# dropping the column Name as it doesn't have impact on prediction
movie_ratings_df.drop('Name',axis=1,inplace=True)

# grouping the multiple values
genre_mean_ratings=movie_ratings_df.groupby('Genre')['Rating'].transform('mean')
movie_ratings_df['Genre_mean_ratings']=genre_mean_ratings

Director_mean_ratings=movie_ratings_df.groupby('Director')['Rating'].transform('mean')
movie_ratings_df['Director_mean_ratings']=Director_mean_ratings

Actor1_mean_ratings=movie_ratings_df.groupby('Actor 1')['Rating'].transform('mean')
movie_ratings_df['Actor 1_mean_ratings']=Actor1_mean_ratings

Actor2_mean_ratings=movie_ratings_df.groupby('Actor 2')['Rating'].transform('mean')
movie_ratings_df['Actor 2_mean_ratings']=Actor2_mean_ratings

Actor3_mean_ratings=movie_ratings_df.groupby('Actor 3')['Rating'].transform('mean')
movie_ratings_df['Actor 3_mean_ratings']=Actor3_mean_ratings

# defining x and y for model where x is indepented var and y is Rating which depending on x
x=movie_ratings_df[['Year','Duration','Votes','Genre_mean_ratings','Director_mean_ratings','Actor 1_mean_ratings','Actor 2_mean_ratings','Actor 3_mean_ratings']]
y=movie_ratings_df['Rating']

# spliting data into training and testing part where 20% is for testing part and remaining 80% is for training part
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

with tab3:
              
       # load the linear regression model 
       model=LinearRegression()
       model.fit(x_train,y_train)
       model_predict=model.predict(x_test)
       st.subheader("Prediction Of Movie Ratings ")
       user_year=st.number_input("Enter Year")
       user_votes=st.number_input("Enter votes")
       user_duration=st.number_input("Enter Duration")
       user_mean_genre_ratings=st.number_input("Enter genre mean ratings")
       user_Director_mean_ratings=st.number_input("Enter Director_mean_ratings")
       user_Actor1_mean_ratings=st.number_input("Enter Actor1_mean_ratings")
       user_Actor2_mean_ratings=st.number_input("Enter Actor2_mean_ratings")
       user_Actor3_mean_ratings=st.number_input("Enter Actor3_mean_ratings")
      
       test=pd.DataFrame({'Year':[user_year],'Duration':[user_duration],'Votes':[user_votes],
                          'Genre_mean_ratings':[user_mean_genre_ratings],
                          'Director_mean_ratings':[user_Director_mean_ratings],
                          'Actor 1_mean_ratings':[user_Actor1_mean_ratings],
                          'Actor 2_mean_ratings':[user_Actor2_mean_ratings],
                          'Actor 3_mean_ratings':[user_Actor3_mean_ratings]})
       if st.button("Movie Ratings Prediction"):
              predicted_value=model.predict(test)
              st.write("Movie Ratings : ",predicted_value[0])
with tab4:       
       # evulating performance of the model
       st.subheader("Performance Metrix")
       st.write("meann square error : ",mean_squared_error(y_test,model_predict))
       st.write("meann absolute error : ",mean_absolute_error(y_test,model_predict))
       st.write("R2 score: ",r2_score(y_test,model_predict))















