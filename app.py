######################################################################
# 1. importing libraries - see requirements.txt for all libraries used
######################################################################
import streamlit as st
#import plotly.express as px
from pycaret.classification import setup, compare_models, pull, save_model, load_model, predict_model, plot_model, evaluate_model, tune_model

from ydata_profiling import ProfileReport
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 
#created file to store variables
from variables import short_automl_desc , short_class_desc, short_profile_desc, short_pycaret_desc

######################################################################
# 2. Build the side bar - split sections to debug any issues
######################################################################

# using OS to check if file is up and use as needed
if os.path.exists('uploaded_dataset.csv'):
    df = pd.read_csv('uploaded_dataset.csv', index_col=None)
else:
    df = pd.DataFrame()

# Set the page 
st.set_page_config(layout="wide", page_title="Conor's AutoML App")
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)



######################################################################
# 3. lets build our Home Page & Navigation
######################################################################

choice =  st.radio('Navigation menu', ['Home','Step1: Upload Data', 'Step2: Data Profiling','Step3: Run AutoML','ML Glossary'],horizontal=True)

if choice == 'Home':
    st.title('Magical AutoML App :magic_wand:')
    st.image(width=200, image=f"https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn.dribbble.com%2Fusers%2F410907%2Fscreenshots%2F2591935%2Fscientist.gif&f=1&nofb=1&ipt=cc4d4b0d731bd638dd9572b9986fb8b021850c61c9f2ff92cbf3b775a40b39d4&ipo=images")
    st.header('Welcome to the AutoML Laboratory :microscope:')
    st.info('This app helps you build a machine learning (ML) models without writing a single line of code.')
    st.subheader('How is this useful?')
    st.info("""
            Well the world has gone AI/ ML mad! This app is a great way to get started with ML, understand your data and build a model quickly. 
            
            Building production Machine Learning models requires a lot of time, high quality data, platform infastructure, effort, and expertise. 
            But with the advent of AutoML, the process has become much easier. 
            AutoML is a process of automating the end-to-end process of applying machine learning to real-world problems.
            This app is designed to make the process of building ML models easier and faster.
            
            """)
    st.subheader('Will they be as good as a machine learning model built by a experienced Data Scientist? :microscope:') 
    st.info("Well no, but they will be pretty good and will be great starting point for understanding your data and making intial predictions.")
    st.subheader('Ok so how do I use this app?')
    st.info("Just follow the steps in the navigation menu - I've maker the steps and in a few clicks you'll have an Machine Learning model trained on histroial data that can provide future predictions.")
    st.subheader('But I dont know what any of this means?')
    st.info('I built this as a sandbox app so play around and see how it works, there is a Learn More tab witrh some helpful info and resources.')
    
    #easter egg 1
    if st.button('DO NOT PRESS THIS BUTTON') == True:
        st.success('You rebel you :wink: You found the ballons button,  I think you are ready to start! :rocket:')
        st.info('Select "Step1: Upload Data" in the Navigation to continue.')
        st.balloons()


    
    st.info("""
            This ML app is a demo of AutoML using streamlit, Ydata_profiling and Pycaret. 

            Note this app can only perform AutoML on classifcation problems in this version, more to come soon!
            
            """)
    st.subheader('created by Conor Curley')
    st.image(width=180,image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia1.tenor.com%2Fimages%2Fa0afeb9cc47a7baf61be453b9a5736b2%2Ftenor.gif%3Fitemid%3D5957952&f=1&nofb=1&ipt=cf528c182da24543a702e83f1b68b0432117d3f21be75f3f1848402db8e10426&ipo=images&clickurl=https%3A%2F%2Ftenor.com%2Fsearch%2Fmagic-gifs')
    st.link_button('Say hello on LinkedIn! :wave:', 'https://www.linkedin.com/in/ccurleyds/')
    
######################################################################
# 4. lets build our Upload data page
######################################################################

if choice == 'Step1: Upload Data':
    st.title('Step 1: Upload your dataset')
    st.image(width=200, image=f'https://c.tenor.com/eUsiEZP1DnMAAAAC/beam-me-up-scotty.gif')
    st.header('Use the file uploader to select your dataset or select from the sample datasets:')
    st.info('This app only supports CSV files for now. If you have a different file type, please convert it to a CSV file before uploading.')
     
    # Add a file uploader to the sidebar:    
    st.info('Option 1: Please select a CSV file type as your dataset.')
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")  

    if uploaded_file :
        # Read the uploaded file:
        df = pd.read_csv(uploaded_file,index_col=None)   
        df_holdout= df[:-10]               
        # Save the uploaded file to a csv file:
        df.to_csv('uploaded_dataset.csv', index=False)
        # Display the dataset:
        st.dataframe(df)
        st.success('Data uploaded successfully!') 
        
    # Add common datasets
    st.info('Option 2: Download a sample dataset to use.')
    if st.button('View sample datasets') == True:

        #titanic dataset
        st.subheader('Titanic Passenger Dataset :ship::')
        st.subheader(':violet[Want to predict who will survive based on the passenger infoformation?]')
        st.info('Info: The Titanic dataset is a historical dataset that includes passenger information like age, gender, passenger class, and survival status from the tragic Titanic shipwreck. This is a classic dataset used to train ML models, to predict if a passenger survived or not. using the passenger information.')
        #st.link_button('Link to dataset source','https://github.com/datasciencedojo/datasets/blob/master/titanic.csv')
        
        with open('titanic_data.csv', 'rb') as f:
            if st.download_button(':violet[Download Titanic CSV :ship:]', f, file_name="titanic_data.csv"): 
                st.success('Titanic dataset downloaded :ship:')

        #telco company dataset
        st.subheader('Vodafone Customer Dataset: :phone:')
        st.subheader(':red[Want to predict who will leave (churn) Vodafone based on customer information?] ')
        st.info('Info: This dataset includes customer information used to predict when a customer leaves/churns. Before you ask, yes Churn is silly business term invented to sound technical.')
        #st.link_button('Link to dataset','https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv')
      
        with open('telco_churn.csv', 'rb') as f: 
            if st.download_button(':red[Download Vodafone Customer CSV :phone:]', f, file_name="telco_churn.csv"): 
                st.success('Vodafone dataset downloaded :mobile_phone:')

        #penguins        
        st.subheader('Penguins Speciies Classification Dataset :penguin:')
        st.subheader(':blue[Want to predict the speicies of Penguin based on some observations?]')
        st.info('Info: This dataset is used to predict penguin species. There are 3 different species of penguins in this dataset, collected from 3 islands in the Palmer Archipelago, Antarctica.')
        #st.link_button('link to dataset','https://github.com/dickoa/penguins/blob/master/data/penguins_lter.csv')
   
        with open('penguins.csv', 'rb') as f: 
            if st.download_button(':blue[Download Penguins CSV]', f, file_name="penguins.csv"): 
                st.success('dPenguin dataset downloaded :penguin:')

    # next steps prompt
    if not df.empty:  
        st.success('A Dataset is uploaded, ready to move to the next step!')
        st.subheader(':rainbow[Great job you have dataset loaded! Select "Data Profiling" in the navigation to continue.]')
    else:    
        st.warning('No dataset uploaded yet. Please upload a dataset to continue.')

     

######################################################################
# 5. lets build our data profiling page
######################################################################

if choice == 'Step2: Data Profiling':
    
    #set up the dataset
    if os.path.exists('uploaded_dataset.csv'):
        df = pd.read_csv('uploaded_dataset.csv', index_col=None)
    # Display the dataset:  
    st.title('Step 2: Data Profiling')
    st.image(width=200,image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia0.giphy.com%2Fmedia%2F9ADoZQgs0tyww%2Fgiphy.gif&f=1&nofb=1&ipt=bbe895f57b94a3eb6cc387f2bc4dd996bf548428356950b16d9f17de07feefaf&ipo=images')
    st.header('Whats Data Profiling?')
    st.info("Data profiling is the process of examining the data available and collecting statistics or informative summaries about that data. The purpose of these statistics is to identify potential issues with the data, such as missing values, outliers, or unexpected distributions.")
    st.info("Let's see what how data looks like!")
    st.dataframe(df)
    if st.button('Generate Data Profile Report') == True:
        #create profile report
        profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
        st.image(width=200, image=f'https://visme.co/blog/wp-content/uploads/2016/04/Header-1200-3.gif')
        #rendering the report in the streamlit app
        st.info('Review your dataset profile:')
        st_profile_report(profile)
        
        st.subheader(':rainbow[Look at you go, you profiled your dataset! select "AutoML" in the navigation to continue.]')

######################################################################
# 6. lets build our Run AutoML page
######################################################################

if choice == 'Step3: Run AutoML':
    st.title('Step 3: Run AutoML')
    st.image(width=200, image='https://i.pinimg.com/originals/cc/32/99/cc3299350f3d91327d4a8922ecae8fb8.gif')

    #set up the data
    if os.path.exists('uploaded_dataset.csv'):
        df = pd.read_csv('uploaded_dataset.csv', index_col=None)

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
    st.info('Follow the auotML process steps below to train your model:')
   
    #Step 1 
    st.info('Step 1: So what do you want to predict?')
    st.write("Note: if you are using the titanic dataset, use Survived column. If you are using the Vodafone Customer dataset, use Churn column. If you are using the Penguins dataset, use Species")  
    
    target = st.selectbox("Select Target Variable", df.columns)
    st.info(f"Our ML model with predict the {target} variable.")

    #Step 2 
    st.info("Step 2: Any columns should be ignored? (Select columns such as Names and individual ID numbers,  don't select your Target Variable here.")
    st.write("Note: if you are using the titanic dataset, you may want to ignore the 'Passenger Id', 'Name', 'Ticket' columns. Similar if you are using the Telco Churn dataset, you may want to ignore the 'Customer ID' column. In the Penguins dataset, you may want to ignore the 'Individual' column.")  
    ignore_list= st.multiselect("Select columns to ignore: ",df.columns)
    # Display the dataset for reference:
    st.dataframe(df)

    st.warning(f"You selected the following columns to be ignored: {ignore_list}")
    
    #Step 3
    st.info("Step 3: Ready to run your model? PRESS THE BUTTON BELOW!")
    if st.button('Train my model baby......Whoosh!!!'):
        setup(df,target=target,fix_imbalance = True, remove_multicollinearity = True, ignore_features= ignore_list)
        setup_df=pull()
        st.info('Figuring out patterns in the data to make preditions...+15 different ML models will be trained. This may take a few minute - go stick the kettle on, my cat has some serious machine learning work to do!')
        st.image(width=200, image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fc.tenor.com%2FdPLWf7LikXoAAAAC%2Ftyping-gif.gif&f=1&nofb=1&ipt=bc9b10d7dbf1c064885a96862c6f4040b6cfe7c6b4e0c777174f662cc93d2783&ipo=images')
        st.info('PyCaret Settings for AutoML')
        st.dataframe(setup_df)
        best_model = compare_models(sort='AUC')
        compare_df = pull()
        st.info("Results are in! Review your model performance below:")
        st.write("Bloody Oath that's an impressive table of ML models! The best model is at the top - Highest AUC score.")
        #renders the best model leaderboard: 
        st.dataframe(compare_df) 
        best_model = tune_model(best_model)
        best_model
        evaluation= evaluate_model(best_model)
        st.write(evaluation)
        save_model(best_model, 'best_model')   
        st.success('You model was trained successfully on your data! We can now use this model to make predictions.')	
        
    if st.button("Optional: What do these performance scores mean? Click here for more info") == True:
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
    
    if st.button('Optional: View best model performance graphs (test_data)') == True:
        with open('best_model.pkl', 'rb') as f: 
            best_model = load_model('best_model')
        # plot feature importance
        st.subheader("Model Performance Figures: (if available - not all models have these features)")
        try : auc_img = plot_model(best_model, plot="auc", display_format="streamlit", save=True)
        except: pass

        try : cm_img = plot_model(best_model, plot = 'confusion_matrix', display_format="streamlit", save=True)
        except: pass

        try : features_img  = plot_model(best_model, plot = 'feature_all', display_format="streamlit", save=True)
        except: pass
        
        try : pipeline_img  = plot_model(best_model, plot = 'pipeline', display_format="streamlit", save=True)
        except: pass

        
        #render the images
        if 'auc_img' not in locals():
            st.warning('No AUC graph available for this model.')
        else:   
            st.subheader('Fig 1 Model performance: AUC curve')
            st.info('AUC graph is very useful when the target variable is binary. It is a measure of how well a binary classification model is able to distinguish between positive and negative classes.')
            
            st.image(auc_img)
        if 'cm_img' not in locals():
            st.warning('No Confusion Matrix available for this model.')
        else:
            st.subheader('Fig 2 Model performance: Confusion Matrix: '  )
            st.info('A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. It allows the visualization of the performance of an algorithm.')
            st.info('Correct Predictions - Top left: True Negative, Bottom Right: True Positive')
            st.info('Incorrect Predictions - Top Right: False Positive, Bottom Left: False Negative')
            st.image(cm_img)
        if 'features_img' not in locals():
            st.warning('No Feature Importance available for this model.')
        else:  
            st.subheader('Fig 3, Feature Importance:' )
            st.info('Feature importance is a technique that assigns a score to input features based on how useful they are at predicting a target variable. The higher the score, the more important the feature is.')
            st.image(features_img)
        
        if 'pipeline_img' not in locals():
            st.warning('No Model Pipeline available for this model.')
        else:  
            st.subheader('Fig 4, Model Pipeline:')
            st.info('The model pipeline is a visual representation of the steps that are taken to preprocess the data and train the model. It shows the different steps that are involved in the process, such as data cleaning, feature engineering, and model training.')
            st.image(pipeline_img)
        


    #Step 4
    st.info("Step 4: Download your trained Model as a Pickle file:")

    with open('best_model.pkl', 'rb') as f: 
        if st.download_button('Download Model', f, file_name="best_model.pkl"): 
            best_model = load_model('best_model')
            st.write(best_model)
            st.success('Model downloaded successfully!')          
            st.balloons()

    #Step 5
    st.info("Step 5: Prepare some new data to test your model:")
    st.write("Now that you have trained your model, you can test it on new data to see how well it performs.")
    
    if os.path.exists('uploaded_dataset.csv'):
        df = pd.read_csv('uploaded_dataset.csv', index_col=None)
        df_holdout= df.tail(10)   

    test=df_holdout.drop(target, axis=1)
    if st.button("View a sub-sample of your data, without a target variable."):
        st.write("This is an example of what your future data would look like. ") 
        st.dataframe(test)

    #Step 5
    st.info("Step 6: Run your model on new data and get some predictions!:")

    if st.button("Alrighty this is the moment of truth - let's punch thoses numbers and predict with our model!") == True:
        with open('best_model.pkl', 'rb') as f: 
            best_model = load_model('best_model')
        st.write('Applying our ML model ...')
        new_prediction = predict_model(best_model, data=test)
        new_prediction['Target_Variable_Actual'] = df_holdout[target]
        st.dataframe(new_prediction)
        st.success('Predictions made successfully! Have a look at the predictions above in the table! (at the end you will see your prediction and actual classification):rocket:')    

    #Step 5
    st.info("Gotten this far? I think you deserve a dance!:")	
    if st.button("Dance button!") == True:
        st.image(width=500,image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fc.tenor.com%2FZAoUo4PquusAAAAC%2Fyou-did-it-congratulations.gif&f=1&nofb=1&ipt=bfeb6b6934c23145f401edd23610973857097a7938a18d49835bc9dbbc30e0f1&ipo=images')
        st.subheader(":rainbow[Congrats you beauty! You built your own Machine learning model without writing a single line of code!]")
        st.balloons()
        st.success('You have completed the AutoML process! You have trained a model and made predictions on new data. You beauty :wink:')

#############################################################################################
## 7. ML Glossary Section
#############################################################################################

if choice == 'ML Glossary':
    st.title('Machine Learning Glossary:')
    st.image(width=200, image='https://i.pinimg.com/originals/cc/32/99/cc3299350f3d91327d4a8922ecae8fb8.gif')
    
    st.subheader('Technology I used to build this app:')
    if st.button("Want to learn more about Pycaret?") == True:
        st.subheader('What is Pycaret?')
        st.info(short_pycaret_desc)
        st.link_button('Pycaret Documentation', 'https://pycaret.gitbook.io/docs')
    
    if st.button("Want to learn more about Streamlit?") == True:
        st.subheader('What is Streamlit?')
        st.info('Streamlit is an open-source app framework for Machine Learning and Data Science projects. It allows you to build interactive web applications with simple Python scripts.')
        st.link_button('Streamlit Documentation', 'https://docs.streamlit.io/get-started')
    
    if st.button("Want to learn more abut Ydata Profiling?") == True:
        st.subheader('What is Ydata Profiling?')
        st.info('Ydata Profiling is an open-source library that generates profile reports from a pandas DataFrame. These reports contain interactive visualizations that allow you to explore your data.')
        st.link_button('Ydata Profiling Documentation', 'https://pypi.org/project/ydata-profiling/')

    st.header('Explain Classification Problems:')
    st.info(short_class_desc)

    st.subheader('Explain AutoML:')
    st.info(short_automl_desc)

    st.subheader('Explain Data Profiling')
    st.info(short_profile_desc)