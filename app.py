
#importing libraries - see requirements.txt for all libraries used
from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
from ydata_profiling import ProfileReport
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 
import pickle
#created file to store variables
from variables import short_automl_desc , short_class_desc, short_profile_desc, short_pycaret_desc

######################################################################
# Build the side bar - split sections to debug any issues
######################################################################


# using OS to check if file is up and use as needed
if os.path.exists('uploaded_file.csv'):
    df = pd.read_csv('uploaded_file.csv', index_col=None)

# Set the page layout
st.set_page_config(layout="wide", page_title="Conor's AutoML App")
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


######################################################################
# lets build our Home Page & Navigation
######################################################################

choice =  st.radio('Navigation menu', ['Home','Step1: Upload Data', 'Step2: Data Profiling','Step3: Run AutoML', 'Step4: Model download','ML Glossary'],horizontal=True)
if choice == 'Home':
    st.title('Magical AutoML App 🚀')
    st.header('Welcome to the AutoML Laboratory :microscope:')
    st.image(width=200, image=f"https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn.dribbble.com%2Fusers%2F410907%2Fscreenshots%2F2591935%2Fscientist.gif&f=1&nofb=1&ipt=cc4d4b0d731bd638dd9572b9986fb8b021850c61c9f2ff92cbf3b775a40b39d4&ipo=images")
    st.info('This app helps you to help you build a machine learning model without writing a single line of code.')
    st.subheader('How is this useful?')
    st.info("""
            Well the world has gone AI/ ML mad! This app is a great way to get started with ML, understand your data and build a model quickly. 
            
            Building production Machine Learning models requires a lot of time, high quality data, platform infastructure, effort, and expertise. 
            But with the advent of AutoML, the process has become much easier. 
            AutoML is a process of automating the end-to-end process of applying machine learning to real-world problems.
            This app is designed to make the process of building ML models easier and faster.
            
            Will they be as good as a handcrafted model by a Data Scientise? 
            Absolutely not, but they will be a great starting point and very useful for understanding your data and making intial predictions.
            """)
    st.subheader('Ok so how do I use this app?')
    st.info("Just follow the steps in the navigation menu - I've maker the steps and in a few clicks you'll have an Machine Learning model trained on histroial data that can provide future predictions.")
    st.subheader('But I dont know what any of this means?')
    st.info('I built this as a sandbox app so play around and see how it works, there is a Learn More tab witrh some helpful info and resources.')
    if st.button('DO NOT PRESS THIS BUTTON') == True:


        st.success('You rebel you :wink: You found the ballons button,  I think you are ready to start! :rocket:')
        st.info('Select "Step1: Upload Data" in the Navigation to continue.')
        st.balloons()


    with st.sidebar:
        # Add a title to the sidebar:
        st.title('AutoML App ')
        st.info("""
                This ML app is a demo of AutoML using streamlit, Ydata_profiling and Pycaret. 

                Note this app can only perform AutoML on classifcation problems in this version, more to come soon!
                
                """)
        st.subheader('created by Conor Curley')
        st.image(width=180,image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia1.tenor.com%2Fimages%2Fa0afeb9cc47a7baf61be453b9a5736b2%2Ftenor.gif%3Fitemid%3D5957952&f=1&nofb=1&ipt=cf528c182da24543a702e83f1b68b0432117d3f21be75f3f1848402db8e10426&ipo=images&clickurl=https%3A%2F%2Ftenor.com%2Fsearch%2Fmagic-gifs')
        st.link_button('Say hello on LinkedIn! :wave:', 'https://www.linkedin.com/in/ccurleyds/')
    
######################################################################
# lets build our upload data page
######################################################################

if choice == 'Step1: Upload Data':
    st.title('Step 1: Upload your dataset')
    st.image(width=200, image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn.dribbble.com%2Fusers%2F24711%2Fscreenshots%2F3886002%2Ffalcon_persistent_connection_2x.gif&f=1&nofb=1&ipt=e9604ffb0fab14883ea25e5f29a643b74f5e70ade391339595da9f2658a59807&ipo=images')
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

     

######################################################################
# lets build our data profiling page
######################################################################

if choice == 'Step2: Data Profiling':
    st.title('Step 2: Data Profiling')
    st.image(width=200,image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fintellipaat.com%2Fblog%2Fwp-content%2Fuploads%2F2015%2F11%2Fe42cce_756b090fe40548eda9148fd5599980bb_mv2.gif&f=1&nofb=1&ipt=4a14437184fea6efddac5c646c2c10b70b1efa2c9767f3bf403ba5f85cc70c31&ipo=images')
    st.header('Whats Data Profiling?')
    st.info("Data profiling is the process of examining the data available and collecting statistics or informative summaries about that data. The purpose of these statistics is to identify potential issues with the data, such as missing values, outliers, or unexpected distributions.")
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
    st.title('Step 3: Run AutoML')
    st.image(width=200, image='https://i.pinimg.com/originals/cc/32/99/cc3299350f3d91327d4a8922ecae8fb8.gif')
    if st.button('First time at this step? Click here for more info') == True:
        st.subheader("1. So I have profiled my data, what next?")
        st.info('Now we train a machine learning model to predict a target variable. This is where the magic happens!')
        st.subheader("2. How does it work?")
        st.info('Depending on ML Algorithm, the model will learn from historical data to create a set of complex rules, this is your model. We then can use this model to make predictions on new data.')

        st.subheader("3. How are we doing this in this app?")
        st.info('We are using Pycaret, an AutoML library that trains and compares multiple machine learning models. Pycaret will train 10+ models and compare them to find the best model for your data. ')
        
        st.subheader("4. What about cleaning my data? It's a mess!")
        st.info('Pycaret will handle missing values, encoding, scaling, and other data preprocessing steps for you. It will also handle class imbalance and multicollinearity.')
        
    st.subheader("Ready to run some AutoML magic?")
    st.info('Follow the steps below to train your model:')
    #create profile report

    st.info('Step 1: So what do you want to predict?')
    target = st.selectbox("Select Target Variable", df.columns)
    st.info("Our ML model with predict this target",target)
    
    st.info("Step 2: Any columns should be ignored? (Select columns such as Names and individual ID numbers,  don't select your Target Variable here.")
    st.write("Note: if you are usign the titanic dataset, you may want to ignore the 'Passenger Id', 'Name', 'Ticket' columns.")
    
    ignore_list= st.multiselect("Select columns to ignore: ",df.columns)
    # Display the dataset for reference:
    st.dataframe(df)

    st.warning("You selected the following columns to be ignored:", ignore_list)
    
    st.info("Step 3: Ready to run your model? PRESS THE BUTTON BELOW!")
    if st.button('Train my moodel baby......Whoosh!!!'):
        setup(df,target=target,fix_imbalance = True, remove_multicollinearity = True, ignore_features= ignore_list)
        setup_df=pull()
        st.info('Pycaret ML settings : 10+ models will be trained and compared. This may take a few minutes.')
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.write("Bloody Oath that's impressive table of ML models!")
        #renders the best model leaderboard: 
        st.dataframe(compare_df) 
        if st.button("What do these performance scores mean? Click here for more info") == True:
            st.subheader("1. What does 'Accuracy' mean?")
            st.info('Accuracy is the ratio of correctly predicted observations to the total observations. It works well only if there are equal number of samples belonging to each class.')
            st.subheader("2. What does 'AUC' mean?")
            st.info('AUC stands for Area Under the Curve. It is used in classification analysis in order to determine which of the used models predicts the classes best.')
            st.info('An excellent model has AUC near to the 1 which means it has good measure of separability. A poor model has AUC near to the 0 which means it has worst measure of separability.')
            st.subheader("3. What does 'Recall' mean?")
            st.info('Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes, it is the ratio of true positive to the sum of true positive and false negative.')
            st.subheader("4. What does 'Precision' mean?")
            st.info('Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. High precision relates to the low false positive rate.')
            st.subheader("5. What does 'Kappa' mean?")
            st.info('Kappa is a statistic that measures inter-rater agreement for qualitative items. It is generally thought to be a more robust measure than simple percent agreement calculation, as Kappa takes into account the possibility of the agreement occurring by chance.')
            st.subheader("6. What does 'MCC' mean?")
            st.info('MCC is a measure of the quality of binary classifications. It returns a value between -1 and 1. A coefficient of 1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction.')
            st.subheader("7. What does 'TT sec' mean?")
            st.info('TT sec is the time taken to train the model.')
        
        best_model 
        save_model(best_model, 'best_model')    
        st.success('You model was trained successfully on your data! We can now use this model to make predictions.')	
        
        # plot feature importance
        st.subheader("Model Performance Figures:")
        auc_img = plot_model(best_model, plot="auc", display_format="streamlit", save=True)
        cm_img = plot_model(best_model, plot = 'confusion_matrix', plot_kwargs = {'percent' : True}, display_format="streamlit", save=True)
        classr_img = plot_model(best_model, plot = 'class_report', display_format="streamlit", save=True)
        features_img  = plot_model(best_model, plot = 'feature_all', display_format="streamlit", save=True)
        
        
        #render the images
        st.subheader('Fig 1 Model performance: AUC curve')
        st.info('AUC graph is very useful when the target variable is binary. It is a measure of how well a binary classification model is able to distinguish between positive and negative classes.')
        st.image(auc_img)

        st.subheader('Fig 2 Model performance: Confusion Matrix: '  )
        st.info('A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. It allows the visualization of the performance of an algorithm.')
        st.info(f'The confusion matrix is a useful tool for understanding the performance of a classification model. It provides a detailed breakdown of the model\'s predictions, allowing you to see where it is making errors and how well it is performing overall.')
        st.image(cm_img)

        st.subheader('Fig 3, Classification Report' )
        st.info('Classification report is used to measure the quality of predictions from a classification algorithm. How many predictions are True and how many are False. More specifically, True Positives, False Positives, True negatives and False Negatives are used to predict the metrics of a classification report.')
        st.image(classr_img)
        
        st.subheader('Fig 4, Feature Importance:' )
        st.info('Feature importance is a technique that assigns a score to input features based on how useful they are at predicting a target variable. The higher the score, the more important the feature is.')
        st.image(features_img)
        

            
######################################################################
# lets build our Model Download page
######################################################################

if choice == 'Step4: Model download':
   
    # Part 1: download model
    st.subheader('Download your Model :download:')	
    
    with open('best_model.pkl', 'rb') as f: 
        if st.download_button('Download Model', f, file_name="best_model.pkl"): 
            st.success('Model downloaded successfully!')
            st.balloons()

    
    # Part 2: Upload model
    st.subheader('Upload a model to make predictions as Pickle file:')
    uploaded_model = st.file_uploader("Upload Model")
    

    if uploaded_model is not None:
        up_model = load_model(uploaded_model)
        st.success("Model uploaded")
        st.write(up_model)

    # Part 3: Upload data to make predictions
    st.subheader('Upload a dataset to make predictions on:')
    uploaded_data= st.file_uploader("Prediction Data")  
    
    if uploaded_data is not None:
        # Read the uploaded file:
        df_pred = pd.read_csv(uploaded_data,index_col=None) 
        columns_pred= st.multiselect("Select columns model needs: ", df_pred.columns)
        data_to_predict = df_pred[columns_pred]
        # Display the dataset:
        st.dataframe(data_to_predict)
        st.success('Data uploaded successfully!') 

    
    
    if st.button('Ok use by model to make predictions...') == True:
        st.write('Predicting...')
        new_prediction = predict_model(up_model, data=data_to_predict)
        st.dataframe(new_prediction)
        st.write("Done!")
        st.balloons()
        st.success('Predictions made successfully!')
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
    
    

  
