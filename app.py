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
st.set_page_config(layout="wide", page_title="Conor's EasyML App")
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


st.title('EasyML App :rocket:')
choice =  st.radio('Navigation menu', ['Starting Point','Step1: Find your data and upload it!', 'Step2: Make me some pretty graphs!','Step3: Machine Learning Time', 'Step4: Predict the Future!','App Build Notes'],horizontal=True)
st.divider()

######################################################################
# 2. lets build our Home page
######################################################################

if choice == 'Starting Point':
    st.subheader('Welcome to my EasyML App! :rocket:')
    st.image(width=400, image=f"https://lh4.googleusercontent.com/-yc4Fn6CZPtBPbRByD33NofqGnKGDrU5yy0t6ukwKKS5BxPLH5mUGLsetAUOtaK4D1oMp7otcLzuyr7khbRvCGvQjRSXJ5kjSbVOi3jbmHIjzHR7PO8mh52BlNgAHfnrViChn3jH5-z8M-A6M5OsK4c")
    st.info("""
        Well hello there :wave: This app helps users build their own machine learning (ML) models without writing a single line of code!
        
        Few clicks and you will have a trained ML model that can make predictions on new data! :rocket:
       
        """)
    
    st.divider()
 
    st.subheader("Who is this App for? 	:game_die:")
    st.info(":black[This app for anyone who is interested in seeing the end-to-end Machine learning process without writing code]")
    st.info(":black[This app has a simple interface for uploading your dataset, profiling your dataset and then running machine learning algorithms to create a model which you can test.]")
    st.info(":black[I hope you enjoy using the app and building your own ML models!]")
    
    st.divider()
    st.subheader("A little more info: :mag_right:")
    expander = st.expander("Why did I build this? :building_construction:")
    expander.write(""" People are often put off Machine Learning due to the complexity but it doesn't have to be that way. Ive always thought of it as a fun puzzle game so hopeully this app can take a liitle of the mystery and seriousness out of it for you.""")
    expander.write(""" Aslo, you only learn my doing and I wanted to learn more about the Streamlit framework for simple web development and the Pycaret AutomMl package as a prototyping tool""")
  
    expander = st.expander("Ok, what is Machine Learning (ML)? What is ML model? :robot_face:")
    expander.write("Machine learning is a branch of artificial intelligence that uses computer algorithms to learn from data and perform tasks that normally require human intelligence.")
    expander.info("An ML model is a set of complex rules that uses historical and current data to predict future outcomes")
    expander = st.expander("What is AutoML and why is it useful? :computer:")
    expander.write("""
            AutoML is a process of automating the end-to-end process of applying machine learning to real-world problems.
            This app is designed to make the process of building ML models easier and faster.
                   
            Building production Machine Learning models requires a lot of time, high quality data, platform infastructure, effort, and expertise. 
            But with the advent of AutoML, the process has become much easier and faster to build basic starter models.
            
            """)
    expander.info("This app used AutoML technology to build your own ML model without writing code.")

    expander = st.expander('Will my models be as good as one built by a experienced Data Scientist? :microscope:') 
    expander.write("Well no, but they will be pretty good and will be great starting point for understanding your data and making intial predictions.")
    
    expander = st.expander('Ok so how do I use this app? :racing_car:')
    expander.write("Follow the steps in the navigation menu and in a few clicks you'll have an Machine Learning model trained on histroial data that can provide future predictions.")
    

    #easter egg 1
    if st.button(':rainbow[DO NOT PRESS THIS BUTTON]') == True:
        st.balloons()
        st.success('You rebel you :wink: You found the ballons button,  I think you are ready to start! :rocket:')
        st.subheader(':rainbow[Select "Step1" in the navigation to continue.] :point_up_2:')
    
    st.divider()    
    st.subheader('created by Conor Curley')
    st.image(width=180,image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia1.tenor.com%2Fimages%2Fa0afeb9cc47a7baf61be453b9a5736b2%2Ftenor.gif%3Fitemid%3D5957952&f=1&nofb=1&ipt=cf528c182da24543a702e83f1b68b0432117d3f21be75f3f1848402db8e10426&ipo=images&clickurl=https%3A%2F%2Ftenor.com%2Fsearch%2Fmagic-gifs')
    st.link_button('Say hello on LinkedIn! :wave:', 'https://www.linkedin.com/in/ccurleyds/')
    
######################################################################
# 4. lets build our Upload data page
######################################################################

if choice == 'Step1: Find your data and upload it!':

    st.subheader('Step1: Find your data and upload it!')
    st.image(width=200, image=f'https://c.tenor.com/eUsiEZP1DnMAAAAC/beam-me-up-scotty.gif')
    st.subheader('Instructions:')
    st.info('Use the file uploader to select your dataset.')
    st.warning('Note this app can only perform AutoML on classifcation problems - predicting 1 or many outcomes so use a dataset that fits this requirement. Functionality to solve other machine learning problems to come soon!')
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
    st.subheader("Have no dataset? Download and use of these sample datasets")

    if st.button('View sample datasets') == True:

        #titanic dataset
        st.subheader('Titanic Passenger Dataset :ship::')
        st.info("This is the Wonderwall of datasets - everyone who knows it, is sick of it but if you've never seen it then it's a banger!. It is a classic dataset used to train ML models, to predict if a passenger survived or not. using the passenger information.")
        #st.link_button('Link to dataset source','https://github.com/datasciencedojo/datasets/blob/master/titanic.csv')
        
        with open('data/titanic_data.csv', 'rb') as f:
            if st.download_button(':violet[Download Titanic CSV :ship:]', f, file_name="titanic_data.csv"): 
                st.success('Titanic dataset downloaded :ship:')

        #telco company dataset
        st.subheader('Vodafone Customer Dataset: :phone:')
        st.info("This is the 'please don't leave me' dataset, used to predict when a customer leaves/churns. Before you ask, yes Churn is silly business term invented to sound technical.")
        #st.link_button('Link to dataset','https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv')
      
        with open('data/telco_churn.csv', 'rb') as f: 
            if st.download_button(':red[Download Vodafone Customer CSV :phone:]', f, file_name="telco_churn.csv"): 
                st.success('Vodafone dataset downloaded :mobile_phone:')

        #penguins        
        st.subheader('Penguins Speciies Classification Dataset :penguin:')
        st.info("This is the 'Which penguin can I steal?' dataset, used to predict the species of penguins based on some observations. There are 3 different species of penguins in this dataset.")
        #st.link_button('link to dataset','https://github.com/dickoa/penguins/blob/master/data/penguins_lter.csv')
   
        with open('data/penguins.csv', 'rb') as f: 
            if st.download_button(':blue[Download Penguins CSV]', f, file_name="penguins.csv"): 
                st.success('Penguin dataset downloaded :penguin:')

    # next steps prompt
    st.divider()
    
    if not df.empty:  
        st.success('A Dataset is uploaded, ready to move to the next step!')
        st.subheader(':rainbow[Great job you have dataset loaded! Select "Step2" in the navigation to continue.] :point_up_2:')
        
    else:    
        st.warning('No dataset uploaded yet. Please upload a dataset to continue.')

     

######################################################################
# 5. lets build our data profiling page
######################################################################

if choice == 'Step2: Make me some pretty graphs!':
    
    #Set up profile report
      
    st.subheader('Step 2: Make me some pretty graphs!')
    st.image(width=400, image=f'https://visme.co/blog/wp-content/uploads/2016/04/Header-1200-3.gif')
    st.subheader('Instructions:')
    st.info(' 1. Click the button and have a gander! :eyes:')

    expander = st.expander("What is Data Profiling?")
    expander.info("Data profiling is the process of examining the data available and collecting statistics or informative summaries about that data. The purpose of these statistics is to identify potential issues with the data, such as missing values, outliers, or unexpected distributions.")
    
    st.divider()
    
    st.markdown("A sample of your dataset is displayed below:")
        # next steps prompt
    if not df.empty:  
        #if os.path.exists('uploaded_dataset.csv'):
        #    df = pd.read_csv('uploaded_dataset.csv', index_col=None)
        st.dataframe(df.head(10))
    else:    
        st.warning('No dataset uploaded yet. Please upload a dataset to continue.')

    #generate the profile report
    if st.button(':blue[Make those pretty graphs for me!]') == True:
        #create profile report
        start_time_pp = time.time()
        profile = ProfileReport(df, title='PProfiling Report', explorative=True)
        
        #rendering the report in the streamlit app
        st.info('Review your dataset profile:')
        st_profile_report(profile)
        st.write('Time taken to create data profile report', round(((time.time() - start_time_pp)/60), 2), 'mins')
        st.subheader(':rainbow[Look at you go, you profiled your dataset! select "Step3" in the navigation to continue.]:point_up_2:')
        
######################################################################
# 6. lets build our Run AutoML page
######################################################################

if choice == 'Step3: Machine Learning Time':
    st.subheader('Step3: Machine Learning Time')
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
        
    st.subheader("Ready to run some AutoML magic?")
    st.info('Follow the auotML process steps below to train your model:')
   
    st.divider()
    #Step 1 
    st.info('1: Selct the target variable - this is the column you want to to predict.')
    st.write("Note: if you are using the titanic dataset, use Survived column. If you are using the Vodafone Customer dataset, use Churn column. If you are using the Penguins dataset, use Species")  
    target = st.selectbox("Select your Target Variable", df.columns)
    st.success(f"Our ML model with predict:  {target}")
    variables.temp_target = target

    #Step 2 
    st.info("2: Select columns that should be ignored:")
    st.write("""Note: if you are using the titanic dataset, you may want to ignore the 'Passenger Id', 'Name', 'Ticket' columns. 
             Similar if you are using the Vodafone dataset, you may want to ignore the 'Customer ID' column. 
             In the Penguins dataset, you may want to ignore the 'Individual' + 'Sample Number' columns.""")
           
    temp_df=df.drop(target, axis=1)
    ignore_list= st.multiselect("Select columns to ignore: ",temp_df.columns)
    # Display the dataset for reference:
    st.dataframe(df.head(10))
    st.warning(f"You selected the following columns to be ignored: {ignore_list}")

    expander = st.expander("Why do we ignore columns?")
    expander.info("Ignoring columns can help improve the accuracy of the model by removing irrelevant or redundant data. This can help the model focus on the most important features in the data.")
    
    st.divider()
    #Step 3

    st.info("3: Ready to run your model? PRESS THE BUTTON BELOW!")

    if st.button(':rainbow[Train my model baby......Whoosh!!!]'):

        setup(df,target=target,fix_imbalance = True, remove_multicollinearity = True, ignore_features= ignore_list,fold=4,normalize = True)
        setup_df=pull()
        start_time = time.time()
        st.image(width=400, image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fc.tenor.com%2FdPLWf7LikXoAAAAC%2Ftyping-gif.gif&f=1&nofb=1&ipt=bc9b10d7dbf1c064885a96862c6f4040b6cfe7c6b4e0c777174f662cc93d2783&ipo=images')
        st.info('Beep bop beep bop  .... go pop the kettle one, this takes a couple mins as my cat crunches those numbers!')
        st.warning("""Note for adavnaced users: 
                    To improve train time, I have simplified the model parameters. I limited to only a few models, added a time budget, sorted by accuracy (more to simplify for new users), removed hyper-parameter tuning, turned on turbo versions of algorithms and reduced Kfold to 4. I think this is a good starting point for most users. If you want to change these settings, you can do so in the code.""")
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
    expander.info('Accuracy is the ratio of correctly predicted observations to the total observations. It works well only if there are equal number of samples belonging to each class.')
    expander.subheader("2. What does 'AUC' mean?")
    expander.info('AUC stands for Area Under the Curve. It is used in classification analysis in order to determine which of the used models predicts the classes best.')
    expander.info('An excellent model has AUC near to the 1 which means it has good measure of separability. A poor model has AUC near to the 0 which means it has worst measure of separability.')
    expander.subheader("3. What does 'Recall' mean?")
    expander.info('Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes, it is the ratio of true positive to the sum of true positive and false negative.')
    expander.subheader("4. What does 'Precision' mean?")
    expander.info('Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. High precision relates to the low false positive rate.')
    expander.subheader("5. What does 'f1' mean?")
    expander.info('F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. It is a good way to show that a classifer has a good value for both false positives and false negatives.')    
    expander.subheader("5. What does 'Kappa' mean?")
    expander.info('Kappa is a statistic that measures inter-rater agreement for qualitative items. It is generally thought to be a more robust measure than simple percent agreement calculation, as Kappa takes into account the possibility of the agreement occurring by chance.')
    expander.subheader("6. What does 'MCC' mean?")
    expander.info('MCC is a measure of the quality of binary classifications. It returns a value between -1 and 1. A coefficient of 1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction.')
    expander.subheader("7. What does 'TT sec' mean?")
    expander.info('TT sec is the time taken to train the model.')
    
    
    expander=st.expander('Optional: I really really want to know more about these models, show me the model performance graphs!')

    #Expander for the model performance
    expander.info('The following graphs show the performance of the best model. These graphs are useful for understanding how well the model is performing and where it can be improved.')
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
# # 7. lets build our Download & Predict with your model page
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
            if st.download_button('Download Model', f, file_name="best_model.pkl"): 
                best_model = load_model('best_model')
                st.markdown(best_model)
                st.success('Model downloaded successfully!')          
                st.balloons()       
            
    else: 
        st.warning('No model available for download.')
   
    expander = st.expander("What is a Pickle file?")
    expander.info("A pickle file is a serialized file that can be saved to disk. It allows you to save your model so that you can use it later without having to retrain it. Pickle files are a common way to save machine learning models.")
    
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
        st.write("This is an example of what your future data would look for new predictions") 
        st.dataframe(test)

    st.divider()
    #Step 5
    st.info("3: Run your model on new data and get some predictions!:")

    if st.button("Alrighty this is the moment of truth - let's punch thoses numbers and predict with our model!") == True:
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
## 7. ML Glossary Section
#############################################################################################

if choice == 'App Build Notes':
    st.subheader('App Build Notes')
    st.image(width=300, image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia.tenor.com%2FUg6cbVA1ZsMAAAAM%2Fdeveloper.gif&f=1&nofb=1&ipt=7285c5bfb06e6eae122b6d0a5d10980b726559be7da57a57848a303739a2738e&ipo=images')
    st.subheader('How I built this app:')
    st.info('I built this app using the Streamlit framework for the web interface and the Pycaret library for the machine learning models.')
    st.markdown("""Some Key Learnings:
            - Package management is a complete headache as you figure it out. This app is on Python 3.10 and I added the absolute minimum list of libraries int the requirements.txt as I could to reduce errors. Pycaret and scikit-learn have a lot of included dependencies so once you have them installed, you are good to go ive learned. I also kept some not pinned to a specific version and let the streamlit package manager avoid conflicts
            - Streamlit is a great tool for building simple web app with Python but it has some limitations. I had to use a lot of workarounds to get the app to work as I wanted.
            - Userflow could be a bit better - trying to strike a balance between simplicity and showcasing the ML functionality was tricky.
            - Hosting the app on streamlit sharing was a bit of a pain - I had to configure the modelling, previous ML training took 24mins! to work with the free tier limitations.
            - I'll use Heroku next time for hosting as it has more flexibility and better performance. 
            - I'll explore flask to compare next and use this app as a base
            - All in all, I learned a lot building this app and I'm excited to build and release more ML apps 
             """)
    st.divider()

    st.subheader('Technology I used to build this app:')
    if st.button("Want to learn more about Pycaret?") == True:
        st.subheader('What is Pycaret?')
        st.info(variables.short_pycaret_desc)
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

    st.subheader('created by Conor Curley')
    st.image(width=180,image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia1.tenor.com%2Fimages%2Fa0afeb9cc47a7baf61be453b9a5736b2%2Ftenor.gif%3Fitemid%3D5957952&f=1&nofb=1&ipt=cf528c182da24543a702e83f1b68b0432117d3f21be75f3f1848402db8e10426&ipo=images&clickurl=https%3A%2F%2Ftenor.com%2Fsearch%2Fmagic-gifs')
    st.link_button('Say hello on LinkedIn! :wave:', 'https://www.linkedin.com/in/ccurleyds/')


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

