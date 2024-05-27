import streamlit as st

st.set_page_config(
    layout='wide',
    initial_sidebar_state="auto",
    menu_items=None)

import pandas as pd
import os
from operator import index

#created file to store variables
from variables import short_automl_desc

# data viz libraries
import plotly.express as px

# ML libraries
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling

#Data profiling
from streamlit_pandas_profiling import st_profile_report
import os 

#nice article here:
#https://www.datacamp.com/tutorial/pandas-profiling-ydata-profiling-in-python-guide


# Streamlit app code
st.title('Automated Machine Learning App 🚀')
st.write('By: Conor Curley')


with st.sidebar:
    # Add an image to the sidebar:
    st.image('https://i.pinimg.com/originals/cc/32/99/cc3299350f3d91327d4a8922ecae8fb8.gif')
    # Add a title to the sidebar:
    st.title('AutoML App Navigation')
    st.info('This ML app is a demo of AutoML using streamlit, pandas, pycaret and scikit-learn')
    # Add a radio button to the sidebar: 
    choice =  st.radio('navigation menu', ['Upload Data', 'Data Profiling','Run AutoML', 'View Results','About AutoML']) 

# using OS to check if file is up and use as needed
if os.path.exists('uploaded_file.csv'):
    df = pd.read_csv('uploaded_file.csv', index_col=None)

# lets build our upload data page
if choice == 'Upload Data':
    
    st.header('Select your dataset:')
    st.info('Please select a CSV file type as your dataset.')
    st.text('')

    # Add a file uploader to the sidebar:    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")   
    df = pd.DataFrame()
    if uploaded_file :
        # Read the uploaded file:
        df = pd.read_csv(uploaded_file,index_col=None)                  
        # Save the uploaded file to a csv file:
        df.to_csv('uploaded_file.csv', index=False)
        # Display the dataset:
        st.dataframe(df)
        st.write(df)
        st.success('Data uploaded successfully!') 
        
    # Add common datasets
    if st.button('Links to sample datasets') == True:
        st.subheader('Titanic Dataset:')
        st.info('Info: The Titanic dataset is a historical dataset that includes passenger information like age, gender, passenger class, and survival status from the tragic Titanic shipwreck.')
        st.write('https://github.com/datasciencedojo/datasets/blob/master/titanic.csv')
        
    # next steps prompt
    if not df.empty:  
        st.subheader('You are ready to roll!')
        st.write('Select "Data Profiling"  in the sidebar to continue.')
    else:    
        st.warning('No dataset uploaded yet. Please upload a dataset to continue.')

             

# lets build our data profiling page
if choice == 'Data Profiling':
    st.title('Automated Data Profiling:')
    st.info("Let's see what how data looks like!")

    
    
    
    st.write('Select "Run AutoML" in the sidebar to continue.')


# lets build our About AutoML page
if choice == 'About AutoML':
    st.title('About AutoML:')
    st.info(short_automl_desc)
    st.image('https://i.pinimg.com/originals/cc/32/99/cc3299350f3d91327d4a8922ecae8fb8.gif')
    st.success('Welcome to the AutoML app!')

  

