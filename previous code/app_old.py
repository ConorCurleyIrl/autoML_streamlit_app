

#importing libraries - see requirements.txt for all libraries used
from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
from ydata_profiling import ProfileReport
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 

#created file to store variables
from variables import short_automl_desc , short_class_desc, short_profile_desc, short_pycaret_desc

######################################################################
# Build the side bar - split sections to debug any issues
######################################################################


# using OS to check if file is up and use as needed
if os.path.exists('uploaded_file.csv'):
    df = pd.read_csv('uploaded_file.csv', index_col=None)

# Set the page layout
st.set_page_config(layout="wide")


######################################################################
# lets build our Home Page & Navigation
######################################################################

with st.sidebar:
    # Add an image to the sidebar:
    st.image('https://i.pinimg.com/originals/cc/32/99/cc3299350f3d91327d4a8922ecae8fb8.gif')
    # Add a title to the sidebar:
    st.title('AutoML App Navigation')
    # Add a radio button to the sidebar: 
    choice =  st.radio('Navigation menu', ['Home','Step1: Upload Data', 'Step2: Data Profiling','Step3: Run AutoML', 'Step4: Model download','ML Glossary']) 

    st.info("This ML app is a demo of AutoML using streamlit, pandas, pycaret and scikit-learn. Note this app can only perform AutoML on classifcation problems in this version, more to come soon! What is a CLassification problem? See 'ML Glossary' selection in the sidebar")


    if choice == 'Home':
        st.title('Magical AutoML App 🚀')
        st.subheader('created by Conor Curley')
        st.image(f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia1.tenor.com%2Fimages%2Fa0afeb9cc47a7baf61be453b9a5736b2%2Ftenor.gif%3Fitemid%3D5957952&f=1&nofb=1&ipt=cf528c182da24543a702e83f1b68b0432117d3f21be75f3f1848402db8e10426&ipo=images&clickurl=https%3A%2F%2Ftenor.com%2Fsearch%2Fmagic-gifs')
        

######################################################################
# lets build our upload data page
######################################################################

if choice == 'Step1: Upload Data':
    
    st.header('Select your dataset:')
    st.info('Please select a CSV file type as your dataset.')
    

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
        #st.success('Data uploaded successfully!') 
        
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

     

######################################################################
# lets build our data profiling page
######################################################################

if choice == 'Step2: Data Profiling':
    st.title('Automated Data Profiling: :bar_chart:')
    st.info("Let's see what how data looks like!")
    if st.button('Generate Data Profile Report') == True:
        #create profile report
        profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
        #rendering the report in the streamlit app
        st.info('Review your dataset profile:')
        st_profile_report(profile)
        st.write('Select "Run AutoML" in the sidebar to continue.')
    



######################################################################
# lets build our Run AutoML page
######################################################################
#ML libraries
import sklearn 
from sklearn.utils import _get_column_indices
from operator import index
from pycaret.classification import *


if choice == 'Step3: Run AutoML':
    st.title('Run Automated Machine learning:')
    st.subheader("Now let's run some AutoML magic!")
    #create profile report

    st.info('Step 1: So what do you want to predict?')
    target = st.selectbox("Select Target Variable", df.columns)
    st.write("Our ML model with predict this target:", target)
    
    st.info('Step 2: Any columns should be ignored? (names, ids, etc - not needed for prediction)')
    ignore_list= st.multiselect("select columns to ignore:",df.columns)
    # Display the dataset:
    st.dataframe(df)
    st.write("You selected the following columns to be ignored:", ignore_list)
    
    st.info("Step 3: Ready to run your model? PRESS THE BUTTON BELOW!")
    if st.button('Train my moodel baby......Whoosh!!!'):
        setup(df,target=target,fix_imbalance = True, remove_multicollinearity = True, ignore_features= ignore_list)
        setup_df=pull()
        st.info('Pycaret ML settings : 10+ models will be trained and compared. This may take a few minutes.')
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.write("Best Model: bloody Oath that's impressive ")
        #renders the best model leaderboard: 
        st.dataframe(compare_df) 
        best_model 
        save_model(best_model, 'best_model')    
        st.success('Model trained successfully!')	
        auc_img = plot_model(best_model, plot="auc", display_format="streamlit", save=True)
        cm_img = plot_model(best_model, plot = 'confusion_matrix', plot_kwargs = {'percent' : True}, display_format="streamlit", save=True)
        classr_img = plot_model(best_model, plot = 'class_report', display_format="streamlit", save=True)

        # plot feature importance
        features_img  = plot_model(best_model, plot = 'feature_all', display_format="streamlit", save=True)
        #render the images
        st.info('Fig 1 Model performance: AUC curve -  this is used to understand model performance.' )
        st.image(auc_img)
        st.info('Fig 2 Model performance: Confusion Matrix: This shows the prediction performance.'  )
        st.image(cm_img)
        st.info('Fig 3, Feature Importance: This shows which features have the most predictive power.' )
        st.image(classr_img)
        
        st.info('Fig 4, Feature Importance: This shows which features have the most predictive power.' )
        st.image(features_img)
        
    

    
            
######################################################################
# lets build our Model Download page
######################################################################

if choice == 'Step4: Model download':
    with open('best_model.pkl', 'rb') as f: 
        if st.download_button('Download Model', f, file_name="best_model.pkl"): 
            st.success('Model downloaded successfully!')
            st.balloons()

######################################################################
# lets build our ML Glossary page
######################################################################

if choice == 'ML Glossary':
    st.title('Machine Learning Glossary:')
    st.image('https://i.pinimg.com/originals/cc/32/99/cc3299350f3d91327d4a8922ecae8fb8.gif')
    
    st.header('Explain Classification Problems:')
    st.info(short_class_desc)

    st.subheader('Explain AutoML:')
    st.info(short_automl_desc)

    st.subheader('Explain Data Profiling')
    st.info(short_profile_desc)
    
    st.subheader('Explain Pycaret Package')
    st.info(short_pycaret_desc)
    
    

  
