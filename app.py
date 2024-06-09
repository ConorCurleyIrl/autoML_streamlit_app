######################################################################
# 1. importing libraries - see requirements.txt for all libraries used
######################################################################y
import streamlit as st
from pycaret.classification import setup, compare_models, pull, save_model, load_model, predict_model, plot_model, evaluate_model, tune_model
import pandas as pd
import time
from streamlit_pandas_profiling import st_profile_report

# ydata_profiling is the latest howerver it is not supported by streamlit_pandas_profiling so using older versio pandas_profiling
#from pandas_profiling.profile_report import ProfileReport
from ydata_profiling import ProfileReport


import os #file management
import gc #garbage collection
import variables #created file to store variables

######################################################################
# 1. Configuring the app
######################################################################

# using OS to check if file is up and use as needed
if os.path.exists('uploaded_dataset.csv'):
    df = pd.read_csv('uploaded_dataset.csv', index_col=None)
else:
    df = pd.DataFrame()

# Set the page 
st.set_page_config(layout="wide", page_title="Conor's EasyML App (1.0)", page_icon=":rocket:")
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


st.title('EasyML App :rocket:')
choice =  st.radio('Navigation menu', ['Starting Point 🏎️','Step1: Find your data and upload it!', 'Step2: Make me some pretty graphs!','Step3: Machine Learning Time', 'Step4: Predict the Future!','App Build Notes'],horizontal=True)
st.divider()

######################################################################
# Home page
######################################################################

if choice == 'Starting Point 🏎️':
    st.subheader('Welcome to my EasyML App! :rocket:')
    st.image(width=400, image=f"https://lh4.googleusercontent.com/-yc4Fn6CZPtBPbRByD33NofqGnKGDrU5yy0t6ukwKKS5BxPLH5mUGLsetAUOtaK4D1oMp7otcLzuyr7khbRvCGvQjRSXJ5kjSbVOi3jbmHIjzHR7PO8mh52BlNgAHfnrViChn3jH5-z8M-A6M5OsK4c")
    st.info("""
        Well hello there :wave: This app helps users build their own Machine Learning (ML) models without writing a single line of code! With a few clicks, you will have a trained ML model that can make predictions on new data! 🚀
        """)
    
    st.divider()
 
    st.subheader("Who is this App for? 	:game_die:")
    st.info(":black[This app is for anyone who is interested in seeing the end-to-end Machine Learning process without writing code]")
    st.info(":black[This app has a simple interface for uploading your dataset, profiling your dataset and then running multiple Machine Learning algorithms to create a predictive model which you can test!]")
    st.info(":black[I hope you enjoy using the app and building your own ML models!]")
    
    st.divider()
    st.subheader("A little more info: :mag_right:")
    expander = st.expander("Why did I build this? :building_construction:")
    expander.write(""" People are often put off Machine Learning due to the complexity but I've always thought of it as a fun puzzle game! Hopefully this app can take a little of the mystery and seriousness out of it!.""")
    expander.write(""" Also, you only learn by doing and I wanted to learn more about the Streamlit framework for simple web development and the Pycaret AutomMl package as a prototyping tool""")
  
    expander = st.expander("Ok, what is Machine Learning (ML)? What is ML model? :robot_face:")
    expander.write("Machine learning is a branch of artificial intelligence that uses computer algorithms to learn from data and perform tasks that normally require human intelligence.")
    expander.info("An ML model is a set of complex rules that uses historical and current data to predict future outcomes")
    expander = st.expander("What is AutoML and why is it useful? :computer:")
    expander.write("""
            AutoML is a process of automating the end-to-end process of applying Machine Learning to real-world problems.
            This app is designed to make the process of building ML models easier and faster.
                   
            Building production Machine Learning models requires a lot of time, high quality data, platform infrastructure, effort, and expertise. 
            But with the advent of AutoML, the process has become much easier and faster to build basic starter models.
            
            """)
    expander.info("This app used AutoML technology to build your own ML model without writing code.")

    expander = st.expander('Will my model be as good as one built by an experienced Data Scientist? :microscope:') 
    expander.write("Well no, but they will be pretty good and will be a great starting point for understanding your data and making initial predictions.")
    
    expander = st.expander('Ok so how do I use this app? :racing_car:')
    expander.write("Follow the steps in the navigation menu and in a few clicks you'll have a Machine Learning model trained on historical data that can provide future predictions.")
    

    #easter egg 1
    if st.button(':rainbow[DO NOT PRESS THIS BUTTON]') == True:
        st.balloons()
        st.success('You rebel you :wink: You found the balloons button,  I think you are ready to start! :rocket:')
        st.subheader(':rainbow[Select "Step1" in the navigation to continue.] :point_up_2:')
    
    st.divider()    
    st.subheader('created by Conor Curley')
    st.image(width=180,image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia1.tenor.com%2Fimages%2Fa0afeb9cc47a7baf61be453b9a5736b2%2Ftenor.gif%3Fitemid%3D5957952&f=1&nofb=1&ipt=cf528c182da24543a702e83f1b68b0432117d3f21be75f3f1848402db8e10426&ipo=images&clickurl=https%3A%2F%2Ftenor.com%2Fsearch%2Fmagic-gifs')
    st.link_button('Say hello on LinkedIn! :wave:', 'https://www.linkedin.com/in/ccurleyds/')
    
######################################################################
# Upload data page
######################################################################

if choice == 'Step1: Find your data and upload it!':

    st.subheader('Step1: Find your data and upload it!')
    st.image(width=200, image=f'https://c.tenor.com/eUsiEZP1DnMAAAAC/beam-me-up-scotty.gif')
    st.subheader('Instructions:')
    st.info('Use the file uploader to select your dataset.')
    st.warning('Note: this app can only perform solve classifIcation problems - predicting 1 or many outcomes, select a dataset that fits this requirement. ')
    st.divider()

    #set up the dataset
    df = pd.DataFrame()
    # Add a file uploader to the sidebar:    
    st.subheader("We can't build a model without data, please select a dataset to use:")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")  

    if uploaded_file :
        # Read the uploaded file:
        df = pd.read_csv(uploaded_file,index_col=None)   
        df_holdout= df[:-10]               
        # Save the uploaded file to a csv file:
        df.to_csv('uploaded_dataset.csv', index=False)
        # Display the dataset:
        st.success('Dataset uploaded successfully! Here is a sample of your dataset:')
        st.dataframe(df.head(100))
        
    
    st.divider()

    # Add common datasets
    st.subheader("Have no dataset? Download and use one of these sample datasets! ")

    if st.button('View sample datasets:') == True:
        st.info("These datasets are open-source and can be used for educational purposes. I've included them in the app for you to use. I have slighyly modified the datasets to make them easier to use in the app - moved the order of coloumns, removed some columns and added some missing values. See App Build Notes for more info and links to original datasets.")

        #titanic dataset
        st.subheader('Titanic Passenger Dataset :ship::')
        st.info("This is the Wonderwall of datasets - everyone who knows it, is sick of it but if you've never seen it then it's a banger!. It is a classic dataset used to train ML models, to predict if a passenger survived or not. using the passenger information.")
        st.image(width=300, image=f"https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FrvhpSE1rQFsnC%2F200.gif&f=1&nofb=1&ipt=61cb70717c6d7e13616619274bbbaf66e471d15d3751d767e03ad3060a91aeff&ipo=images")
    
        with open('data/titanic_data.csv', 'rb') as f:
            if st.download_button(':violet[Download Titanic CSV :ship:]', f, file_name="titanic_data.csv"): 
                st.success('Titanic dataset downloaded :ship:')

        #telco company dataset
        st.subheader('Vodafone Customer Dataset: :phone:')
        st.info("This is the 'please don't leave me' dataset, used to predict when a customer leaves/churns. Before you ask, yes Churn is silly business term invented to sound technical.")
        st.image(width=300, image=f"https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fc.tenor.com%2F8FKTJDMvH2IAAAAC%2Fhomer-simpsons.gif&f=1&nofb=1&ipt=0dcde120abcd6aa2f5a6520116b46ccbeeef91232629975183ddbbbda791cb2f&ipo=images")
        with open('data/telco_churn.csv', 'rb') as f: 
            if st.download_button(':red[Download Vodafone Customer CSV :phone:]', f, file_name="telco_churn.csv"): 
                st.success('Vodafone dataset downloaded :mobile_phone:')

        #penguins        
        st.subheader('Penguins Species Classification Dataset :penguin:')
        st.info("This is the 'Which penguin can I steal?' dataset, used to predict the species of penguins based on some observations. There are 3 different species of penguins in this dataset.")
        st.image(width=300, image=f"https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.icegif.com%2Fwp-content%2Fuploads%2Fpenguin-icegif.gif&f=1&nofb=1&ipt=977822bbd12a1a908ec2cabb050ff39e04c1a3628e0e8a3ce66799be3cf57a35&ipo=images")
  
        with open('data/penguins.csv', 'rb') as f: 
            if st.download_button(':blue[Download Penguins CSV]', f, file_name="penguins.csv"): 
                st.success('Penguin dataset downloaded :penguin:')


        #Mushroom dataset
        st.subheader('Mushroom Classification Dataset :mushroom:')
        st.info("This is the 'Should Mario eat this?' dataset, used to predict if a mushroom is edible or poisonous based on some observations. Also this has +60,000 rows so it's a big one!")
        st.image(width=300, image=f"https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fi.pinimg.com%2Foriginals%2F4d%2F4c%2Ffc%2F4d4cfc0fa82e58789f811bda40414bc0.gif&f=1&nofb=1&ipt=d338a8039bb3d70ae5d8198661e3f7da03bae8417b9b4cae095e11841301b9c5&ipo=images")
       
        with open('data/mushroom_dataset.csv', 'rb') as f: 
            if st.download_button(':green[Download Mushroom CSV]', f, file_name="mushroom_dataset.csv"): 
                st.success('Mushroom dataset downloaded :mushroom:')



    # next steps prompt
    st.divider()
    
    if not df.empty:  
        st.success('A Dataset is uploaded, ready to move to the next step!')
        st.subheader(':rainbow[Great job you have a dataset loaded! Select "Step2" in the navigation to continue.] :point_up_2:')
        
    else:    
        st.warning('No dataset uploaded yet. Please upload a dataset to continue.')

     

######################################################################
# Data profiling page
######################################################################

if choice == 'Step2: Make me some pretty graphs!':
    
    #Set up profile report
      
    st.subheader('Step 2: Make me some pretty graphs!')
    st.image(width=400, image=f'https://visme.co/blog/wp-content/uploads/2016/04/Header-1200-3.gif')
    st.subheader('Instructions:')
    st.info(' 1. Click the button down the page and have a gander at the report! :eyes:')

    expander = st.expander("What is Data Profiling?")
    expander.info("Data profiling is the process of examining the data available and collecting statistics or informative summaries about that data. The purpose of these statistics is to identify potential issues with the data, such as missing values, outliers, or unexpected distributions.")
    
    st.divider()
    
    st.markdown("A sample of your dataset is displayed below:")
        # next steps prompt
    if not df.empty:  
        #if os.path.exists('uploaded_dataset.csv'):
        #    df = pd.read_csv('uploaded_dataset.csv', index_col=None)
        st.info('This datsaset has ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.')
        st.dataframe(df.head(10))
    else:    
        st.warning('No dataset uploaded yet. Please upload a dataset to continue.')

    #generate the profile report
    if st.button(':blue[Make those pretty graphs for me!]') == True:
        #create profile report
        start_time_pp = time.time()
        profile = ProfileReport(df, title='Profiling Report', explorative=True)
        
        #rendering the report in the streamlit app
        st.info('Review your dataset profile:')
        st_profile_report(profile)
        st.write('Time taken to create data profile report', round(((time.time() - start_time_pp)/60), 2), 'mins')
        st.subheader(':rainbow[Look at you go, you profiled your dataset! select "Step3" in the navigation to continue.]:point_up_2:')
        
######################################################################
# Machine Learning page
######################################################################

if choice == 'Step3: Machine Learning Time':
    st.subheader('Step3: Machine Learning Time ')
    st.image(width=400, image='https://i.pinimg.com/originals/cc/32/99/cc3299350f3d91327d4a8922ecae8fb8.gif')
    st.subheader('Instructions:')
    st.info('1. Select the target variable(column) you want to predict.')
    st.info('2. Select the columns to ignore, removes irrelevant columns.')
    st.info('3. Hit that train model button and watch the magic happen!.')
    st.warning('Optional: Review the model performance statistics and graphs.')
   

    st.divider()
    #set up the data
    if os.path.exists('uploaded_dataset.csv'):
        df = pd.read_csv('uploaded_dataset.csv', index_col=None)


    expander= st.expander('First time at this step? Click here for more info')
    expander.subheader("1. So I have profiled my data, what next?")
    expander.info('Now we train a machine learning model to predict a target variable. This is where the magic happens!')
    expander.subheader("2. How does it work?")
    expander.info('Depending on ML Algorithm, the model will learn from historical data to create a set of complex rules, this is your model. We then can use this model to make predictions on new data.')

    expander.subheader("3. How are we doing this in this app?")
    expander.info('We are using Pycaret, an AutoML library that trains and compares multiple machine learning models. Pycaret will train 10+ models and compare them to find the best model for your data. ')
        
    expander.subheader("4. What about cleaning my data? It's a mess!")
    expander.info('Pycaret will handle missing values, encoding, scaling, and other data preprocessing steps for you. It will also handle class imbalance and multicollinearity.')
        
    st.subheader("Ready to run some Machine Learning magic?")
    st.info('Follow the process steps below to train your model:')
   
    st.divider()
    #Step 1 
    st.info('1: Selct the target variable - this is the column you want to to predict.')
    st.write("Note: if you are using the titanic dataset, use Survived column. If you are using the Vodafone Customer dataset, use Churn column. If you are using the Penguins dataset, use Species")  
    target = st.selectbox("Select your Target Variable", df.columns)
    st.success(f"Our ML model will predict this column:  {target}")
    variables.temp_target = target

    #Step 2 
    st.info("2: Select columns that should be ignored:")
    st.markdown("""Note:
                - Generally, Select columns that are IDs, have significant missing values or are irrelevant to the model.""")

    st.markdown("- If you are using the titanic dataset, you may want to ignore the 'Passenger Id','Cabin_Num' 'Name', 'Ticket' columns. :ship:")
    st.markdown("- Similar if you are using the Vodafone dataset, you may want to ignore the 'Customer ID' column.  :phone:")
    st.markdown("- In the Penguins dataset, you may want to ignore the 'Individual' + 'Sample Number' columns. :penguin:")
    st.markdown("- In the Mushroom dataset, you are good to go! :mushroom:")
            
           
    temp_df=df.drop(target, axis=1)
    
    ignore_list= st.multiselect("Select columns to ignore: ",temp_df.columns)
    expander.success("Note: Ignoring columns is optional and depends on the dataset. If you are unsure, you can leave this blank.")
    # Display the dataset for reference:
    st.dataframe(df.head(10))
    st.warning(f"You selected the following columns to be ignored: {ignore_list}")

    expander = st.expander("Why do we ignore columns?")
    expander.info("Ignoring columns can help improve the accuracy of the model by removing irrelevant or redundant data. This can help the model focus on the most important features in the data.")
    expander.success("Want to see the impsct of ignoring columns? Run the model with and without the columns and compare the results.")

    st.divider()
    #Step 3

    st.info("3: Ready to run your model? PRESS THE RED BUTTON BELOW!")

    if st.button(':white[Train my model baby......Whoosh!!! :rocket:]', type='primary'):
    #Pycaret setup and model training settings - each line is a setting for the model
    #try except block to handle errors if the dataset is too small or has other issues
        try: 
            setup(df,target=target,
                   fix_imbalance = True,
                   remove_multicollinearity = True, 
                   remove_outliers = True,
                  # numeric_features = True,
                  # categorical_features=True,
                  # low_variance_threshold = 0.1,
                   #multicollinearity_threshold=0.8,
                   ignore_features= ignore_list, 
                   fold=3,
                   normalize = True)
        except: 
            st.warning('There was an unexpected error - have you selected the correct target variable? The target variable needs to be a classification column with 1 or more outcomes.')
            st.warning('Please check the dataset and try again. Perhaps the dataset is too small or has other issues.')
            st.stop()

        
        setup_df=pull()
        start_time = time.time()
        st.image(width=400, image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fc.tenor.com%2FdPLWf7LikXoAAAAC%2Ftyping-gif.gif&f=1&nofb=1&ipt=bc9b10d7dbf1c064885a96862c6f4040b6cfe7c6b4e0c777174f662cc93d2783&ipo=images')
        st.info('Beep bop beep bop  .... pop the kettle on :teapot: this may take a couple mins as my cat crunches those numbers and builds you a model!')
        st.warning('Note: The time taken to train the model will depend on the size of the dataset (the mushroom dataset is a big one) and the complexity of the model.')
        st.warning("""Note for advanced users: 
                - to improve train time, I have simplified the model parameters. I limited to only a few models, added a time budget, sorted by accuracy (more to simplify for new users), removed hyper-parameter tuning, turned on turbo versions of algorithms and reduced Kfold to 4. I think this is a good starting point for most users. If you want to change these settings, you can do so in the code.""")
        exp1 =st.expander('Optional: Want to see the technical settings under the hood?')
        exp1.write('The Pycaret configuration settings used to train the model:')
        exp1.dataframe(setup_df)
        #train the model
        best_model = compare_models(budget_time=2, include=['lr', 'knn', 'nb', 'ridge', 'rf'], sort='Accuracy', turbo=True) 
        compare_df = pull()
        st.success("Bloody Oath that's an impressive table of ML models! The best model is at the top of the leaderboard.")
        #renders the best model leaderboard: 
        st.dataframe(compare_df) 
        
        compare_df.to_csv('results_table.csv', index=False)
        st.write('Time taken to train the model:', round(((time.time() - start_time)/60), 2), 'mins')
        #might review in v2
        best_model= tune_model(best_model)
        st.success('Specs on the best model are below:')
        st.info(best_model)
        
        save_model(best_model, 'best_model')   
        st.success('You model was trained successfully on your data! We can now use this model to make predictions.')
        st.image(width=200, image=f'https://gifdb.com/images/high/borat-very-nice-fist-pump-wq6v07qb55osz1jt.gif')
        st.subheader(':rainbow[Very Nice!, you have trained your machine learning model!! Go to the navigation to continue.]:point_up_2:')
        
    
    st.divider()

    st.info('Optional: Review the model performance statistics and graphs below.')
    
    #load the results table
    try :
        res_temp = pd.read_csv('results_table.csv', index_col=None)
        st.subheader('Model Performance Table:')
        st.dataframe(res_temp) 
    except: pass


    
    expander = st.expander("Optional: So what do these performance scores mean? Click here for more info")
    expander.subheader("Performance Scores:")   

    expander.subheader("1. What does 'Accuracy' mean?")
    expander.success('Short answer - % the model predicted correctly - the higher the better!')
    expander.subheader("2. What does 'AUC' mean?")
    expander.success('Short answer - Area under the curve - the higher the better!')
    expander.info('An excellent model has AUC near to the 1 which means it has good measure of separability. A poor model has AUC near to the 0 which means it has worst measure of separability.')
    expander.subheader("3. What does 'Recall' mean?")
    expander.success("short answer - % of actual positives correctly predicted - the higher the better!")
    expander.subheader("4. What does 'Precision' mean?")
    expander.success('Short answer - % of predicted positives that are actually positive - the higher the better!')
    expander.subheader("5. What does 'f1' mean?")
    expander.success('Short answer - the weighted average of Precision and Recall - the higher the better!')
    expander.subheader("5. What does 'Kappa' mean?")
    expander.success('its the 10th letter of the Greek alphabet')
    expander.info('Kappa is a statistic that measures inter-rater agreement for qualitative items. It is generally thought to be a more robust measure than simple percent agreement calculation, as Kappa takes into account the possibility of the agreement occurring by chance.')
    expander.subheader("6. What does 'MCC' mean?")
    expander.success('Melbourne Cricket Club?')
    expander.info('Matthews correlation coefficient is a measure of the quality of binary classifications. It returns a value between -1 and 1. A coefficient of 1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction.')
    expander.subheader("7. What does 'TT sec' mean?")
    expander.success('A sporty audi')
    expander.info('TT sec is the time taken to train the model.')
    

    expander=st.expander('Optional: I just love this stuff so show me more model performance graphs!')

    #Expander for the model performance
    expander.info('The following graphs show the performance of the best model. These graphs are useful for understanding how well the model is performing and where it can be improved.')
    expander.subheader('These are the performance graphs for the best model based on a unseen holdout set of data used to test how the model would perform on new data.')
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
        expander.warning('No AUC graph available for this model.')
    else:   
        expander.subheader('Fig 1 Model performance: AUC curve')
        expander.info('AUC graph is very useful when the target variable is binary. It is a measure of how well a binary classification model is able to distinguish between positive and negative classes.')
        
        expander.image(auc_img)
    if 'cm_img' not in locals():
        expander.warning('No Confusion Matrix available for this model.')
    else:
        expander.subheader('Fig 2 Model performance: Confusion Matrix: '  )
        expander.info('A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. It allows the visualization of the performance of an algorithm.')
        expander.info('Correct Predictions - Top left: True Negative, Bottom Right: True Positive')
        expander.info('Incorrect Predictions - Top Right: False Positive, Bottom Left: False Negative')
        expander.image(cm_img)
    if 'features_img' not in locals():
        expander.warning('No Feature Importance available for this model.')
    else:  
        expander.subheader('Fig 3, Feature Importance:' )
        expander.info('Feature importance is a technique that assigns a score to input features based on how useful they are at predicting a target variable. The higher the score, the more important the feature is.')
        expander.image(features_img)
    
    if 'pipeline_img' not in locals():
        expander.warning('No Model Pipeline available for this model.')
    else:  
        expander.subheader('Fig 4, Model Pipeline:')
        expander.info('The model pipeline is a visual representation of the steps that are taken to preprocess the data and train the model. It shows the different steps that are involved in the process, such as data cleaning, feature engineering, and model training.')
        expander.image(pipeline_img)

        
######################################################################
#  Download & Predict with your model page
######################################################################



if choice == 'Step4: Predict the Future!':
    st.subheader('Step4: Predict the Future! :rocket:')
    st.image(width=400, image='https://media1.giphy.com/media/ZhESFK96NxbuO1yDgy/giphy.gif')
    st.subheader('Instructions:')
    st.info("1. Let's Download the model.")
    st.info('2. Prepare some new data to test your model.')
    st.info('3. Make predictions. :rocket:')

    st.divider()
    
    
    #Step 1
    st.info("1: Download your trained Model as a Pickle file:")
    #if 'best_model' not in locals():
    if os.path.exists('best_model.pkl'): 
        with open('best_model.pkl', 'rb') as f: 
            if st.download_button('Download Model - (Optional)', f, file_name="best_model.pkl"): 
                best_model = load_model('best_model')
                st.markdown(best_model)
                st.success('Model downloaded successfully!')          
                st.balloons()       
            
    else: 
        st.warning('No model available for download.')
   
    expander = st.expander("What is a Pickle file (.pkl)?")
    expander.info("A pickle file is standardized file type which holds your model pipeline information and the ML model you trained. It allows you to save your model so that you can use it later without having to retrain it.")
    expander.info("You can load the model back into memory using the load_model() function in Pycaret. It is a binary file so you can't open it in a text editor.")
    expander.info("Do you need this file? Not really, but it's a good idea to save it in case you want to use the model later.")
    st.divider()

    #Step 5
    st.info("2: Prepare some new data to test your model:")
    
    st.write("Now that you have trained your model, you can test it on new data to see how well it performs.")
    #yes if anyone is reading this, I know I should have split out the holdout data earlier in the process, but this is a MVP so I'm doing it here.
    
    #temp_target_var, yes I know I should have used a class or function to store these variables, but this is a MVP so I'm doing it here.
    tt= variables.temp_target

    if os.path.exists('uploaded_dataset.csv'):
        df = pd.read_csv('uploaded_dataset.csv', index_col=None)
        df_holdout= df.sample(n=10, random_state=42)

    test=df_holdout.drop(tt, axis=1)
    if st.button("View a sub-sample of your data, without a target variable."):
        st.write("This is an example of what your future data would look like for new predictions") 
        st.dataframe(test)

    st.divider()
    #Step 5
    st.info("3: Run your model on new data and get some predictions!:")

    if st.button("Alrighty this is the moment of truth - let's punch those numbers and predict with our model!") == True:
        with open('best_model.pkl', 'rb') as f: 
            best_model = load_model('best_model')
        st.write('Applying our ML model ...')
        new_prediction = predict_model(best_model, data=test)
        new_prediction['Target_Variable_Actual'] = df_holdout[tt]
        st.dataframe(new_prediction)
        st.success('Predictions made successfully! Have a look at the predictions above in the table! (at the end you will see your prediction and actual classification):rocket:')    

    #easter egg
    st.info("Gotten this far? I think you deserve a dance!:")	
    if st.button("Dance button!") == True:
        st.image(width=500,image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fc.tenor.com%2FZAoUo4PquusAAAAC%2Fyou-did-it-congratulations.gif&f=1&nofb=1&ipt=bfeb6b6934c23145f401edd23610973857097a7938a18d49835bc9dbbc30e0f1&ipo=images')
        st.subheader(":rainbow[Congrats you beauty! You built your own Machine learning model without writing a single line of code! ]:wink:")
        st.info("I hope you enjoyed the app and learned something new. If you have any questions or feedback, please let me know. I'm always happy to help!")
        st.balloons()
        

#############################################################################################
## App Build Notes
#############################################################################################

if choice == 'App Build Notes':
    st.subheader('App Build Notes')
    st.image(width=300, image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia.tenor.com%2FUg6cbVA1ZsMAAAAM%2Fdeveloper.gif&f=1&nofb=1&ipt=7285c5bfb06e6eae122b6d0a5d10980b726559be7da57a57848a303739a2738e&ipo=images')
    st.subheader('How I built this app:')
    st.info('I built this app using the Streamlit framework for the web interface and the Pycaret library for the machine learning models.')
    st.subheader("Some Notes & Learnings: (9-June-2024)")
    st.markdown("- Well its now live and I'm pretty happy with how it turned out. Learned a hell of alot about streamlit, pycaret and more python in general")
    st.markdown("- Few issues getting the package management right, 132 packages currently are needed. This app is now on Python 3.10 (using other versions created package conflicts) and I added the absolute minimum list of libraries in the requirements.txt so I could to reduce errors then use Pycaret and scikit-learn which have a lot of included dependencies so once you have them installed, you are good to go I've learned after many hours debugging. I also kept some libraries not pinned to a specific version and let the streamlit package manager avoid conflicts.")
    st.markdown("- Streamlit is a great tool for building simple web app with Python but it has some limitations - an example is that there is no multi-page functionality. I had to use a lot of workarounds (navigation menu) to get the app to work as I wanted but now I've the hnag of it so I think I can build more complex apps in the future.")
    st.markdown("- Userflow could be a bit better - trying to strike a balance between simplicity and showcasing the ML functionality was tricky. Ideally you would have full ML model training settings available in a sidebar to allow full customisation but I wanted to keep it simple for new users. I may add this in the future.")
    st.markdown("- Next time, I would plan out the app UX structure better next time - I had to refactor a lot of code as I went along to make it more user-friendly.")
    st.markdown("- Hosting the app on streamlit's public infastructure was tricky with the ML modelling need for compute. I had to configure and simplify the modelling process, previous ML training took 24mins! to work with the free tier limitations. But hey its free so I can't complain too much. Let's see how it goes with more users.")
    st.markdown("- I may deploy on Heroku next time for hosting as it has more flexibility and better performance. ")
    st.markdown("- All in all, I learned a lot building this app and I'm excited to build and release more ML apps.")
    st.markdown("- Thanks for checking out the app! If you have any feedback or questions, please let me know via LinkedIn.")
    
    st.divider()

    st.subheader('Technology I used to build this app:')
    if st.button("Want to learn more about Pycaret?") == True:
        st.subheader('What is Pycaret?')
        st.info('Pycaret is an open-source, low-code machine learning library in Python that automates the end-to-end machine learning process. It allows you to train, test, and deploy machine learning models without writing code.')
        st.link_button('Pycaret Documentation', 'https://pycaret.gitbook.io/docs')
    
    if st.button("Want to learn more about Streamlit?") == True:
        st.subheader('What is Streamlit?')
        st.info('Streamlit is an open-source app framework for Machine Learning and Data Science projects. It allows you to build interactive web applications with simple Python scripts.')
        st.link_button('Streamlit Documentation', 'https://docs.streamlit.io/get-started')
    
    if st.button("Want to learn more abut Ydata Profiling?") == True:
        st.subheader('What is Ydata Profiling?')
        st.info('Ydata Profiling is an open-source library that generates profile reports from a pandas DataFrame. These reports contain interactive visualizations that allow you to explore your data.')
        st.link_button('Ydata Profiling Documentation', 'https://pypi.org/project/ydata-profiling/')

    st.divider()

    st.subheader('Links to some datasets used in this app:')
    st.info("These datasets are open-source and can be used for educational purposes. I've included them in the app for you to use. I have slighyly modified the datasets to make them easier to use in the app - moved and renamed coloumns, removed some columns and added some missing values.")
    st.link_button('Link to titanic dataset :ship:','https://github.com/datasciencedojo/datasets/blob/master/titanic.csv')
    st.link_button('link to penguin dataset :penguin: ','https://github.com/dickoa/penguins/blob/master/data/penguins_lter.csv')
    st.link_button('Link to telco dataset :phone:','https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv')
    st.link_button('link to mushroom dataset :mushroom:','https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset')
    

    st.divider()    
    st.subheader('created by Conor Curley')
    st.image(width=180,image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia1.tenor.com%2Fimages%2Fa0afeb9cc47a7baf61be453b9a5736b2%2Ftenor.gif%3Fitemid%3D5957952&f=1&nofb=1&ipt=cf528c182da24543a702e83f1b68b0432117d3f21be75f3f1848402db8e10426&ipo=images&clickurl=https%3A%2F%2Ftenor.com%2Fsearch%2Fmagic-gifs')
    st.link_button('Say hello on LinkedIn if you enjoyed the app! :wave:', 'https://www.linkedin.com/in/ccurleyds/')


################################################
#GARBAGE COLLECTION - remove variables and files
#################################################
#garbage collection

gc.collect()

try : os.remove('results_table.csv') #deletes CSV
except: pass

try : os.remove('uploaded_data.csv') #deletes CSV
except: pass

#try : os.remove('best_model.pkl') #deletes CSV
#except: pass

try : os.remove('AUC.png') #deletes CSV
except: pass

try : os.remove('confusion_matrix.png') #deletes CSV
except: pass

try : os.remove('feature_importance.png') #deletes CSV
except: pass

try : os.remove('pipeline.png') #deletes CSV
except: pass

