######################################################################


def model():
    
    #importing libraries - see requirements.txt for all libraries used
    ######################################################################y
    import streamlit as st
    from pycaret.classification import interpret_model, setup, compare_models, pull, save_model, load_model, predict_model, plot_model, evaluate_model, tune_model
    import pandas as pd
    import time
    from streamlit_pandas_profiling import st_profile_report

    # ydata_profiling is the latest howerver it is not supported by streamlit_pandas_profiling so using older versio pandas_profiling
    #from pandas_profiling.profile_report import ProfileReport
    from ydata_profiling import ProfileReport
    import seaborn as sns
    import matplotlib.pyplot as plt

    #cache the data
    @st.cache_resource(max_entries=10, ttl=3600)
    def load_data():
        return None

    #enable session state
    if 'session_state' not in st.session_state:
        st.session_state.session_state = load_data()


    ######################################################################
    # Machine Learning page
    ######################################################################


    st.subheader('Step 3: Machine Learning Time - Let the magic happen! :mage:')
    st.image(width=300, image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn.dribbble.com%2Fusers%2F836931%2Fscreenshots%2F4841254%2Fcrm.gif&f=1&nofb=1&ipt=78209b38c8be610ce07b8bd2c053adea890a56da4318b663f28ae54f6e1e3fb9&ipo=images')
    st.subheader('Instructions:')
    st.info('1. Select the target variable(column) you want to predict.')
    st.info('2. Select the columns to ignore, removes irrelevant columns.')
    st.info('3. Hit that train model button and watch the magic happen!.')

    st.divider()
    #set up the data

    #Step 1 
    st.info('1: Selct the target variable - this is the column you want to to predict.')
    st.write("Note: if you are using the titanic dataset, use Survived column. If you are using the Vodafone Customer dataset, use Churn column. If you are using the Penguins dataset, use Species")  
    st.session_state.target = st.selectbox("Select your Target Variable", st.session_state.df.columns)
    st.success(f"Our ML model will predict this column:  {st.session_state.target}")

    st.divider()
    #Step 2 
    st.info("2: Select columns that should be ignored:")
    expander = st.expander("Why do we ignore columns?")
    expander.write("Ignoring columns can help improve the accuracy of the model by removing irrelevant or redundant data. This can help the model focus on the most important features in the data.")
    expander.write("Want to see the impsct of ignoring columns? Run the model with and without the columns and compare the results.")
    expander.write("Ignoring columns is optional and depends on the dataset. If you are unsure, you can leave this blank.")
    st.markdown("""
                Ignore columns that are IDs or names, have significant missing values or are irrelevant/redunadant to the model prediction.""")

    st.markdown("- If you are using the titanic dataset, you may want to ignore the 'Passenger Id','Cabin_Num' 'Name', 'Ticket' columns. :ship:")
    st.markdown("- Similar if you are using the Vodafone dataset, you may want to ignore the 'Customer ID' column.  :phone:")
    st.markdown("- In the Penguins dataset, you may want to ignore the 'Individual' + 'Sample Number' columns. :penguin:")
    st.markdown("- In the Mushroom dataset, you are good to go! :mushroom:")

    st.session_state.temp_df=st.session_state.df.drop(st.session_state.target, axis=1)

    ignore_list= st.multiselect("Select columns to ignore: ",st.session_state.temp_df.columns)
    # Display the dataset for reference:
    st.dataframe(st.session_state.df.head(10))
    st.warning(f"You selected the following columns to be ignored: {ignore_list}")


    st.divider()
    #Step 3

    st.info("3: Ready to run your model? PRESS THE RED BUTTON! :rocket:")

    ######################################################################
    #Note: Pycaret handles the data split for you, so you don't need to do it manually.
    #However, we need a sample to show future data for predictions.:
    from sklearn.model_selection import train_test_split 

    st.session_state.X= st.session_state.temp_df
    st.session_state.y=st.session_state.df.drop(st.session_state.temp_df.columns, axis=1)
    
    # using the train test split function 
    X_train, X_test, y_train, y_test = train_test_split(st.session_state.X,st.session_state.y , 
                                    random_state=42,  
                                    test_size=0.05,  
                                    shuffle=True) 

    st.session_state.df_input= pd.concat([X_train, y_train], axis=1)
    st.session_state.df_holdoutX= X_test
    st.session_state.df_holdouty= y_test

    ##################################################################

    if st.button('Crunch those numbers...Whoosh!!! :rocket:', type='primary'):
    #Pycaret setup and model training settings - each line is a setting for the model
    #try except block to handle errors if the dataset is too small or has other issues
        try: 
            setup(st.session_state.df_input,target=st.session_state.target,
                    fix_imbalance = True,
                    remove_multicollinearity = True, 
                    remove_outliers = True,
                    #numeric_features = True,
                    #categorical_features=True,
                    # low_variance_threshold = 0.1,
                    multicollinearity_threshold=0.8,
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
        st.write('Beep bop beep bop  .... pop the kettle on :teapot: this may take a couple mins as my cat crunches those numbers and builds you several different models to compare performance!')
        exp1 =st.expander('Optional: Want to see the technical settings under the hood?')
        exp1.write('The Pycaret configuration settings used to train the model:')
        exp1.write('Note: The time taken to train the model will depend on the size of the dataset (the mushroom dataset is a big one) and the complexity of the model.')
        exp1.write("""Note for advanced users: 
                - to improve train time, I have simplified the model parameters. I limited to only a few models, added a time budget, sorted by accuracy (more to simplify for new users), removed hyper-parameter tuning, turned on turbo versions of algorithms and reduced Kfold to 4. I think this is a good starting point for most users. If you want to change these settings, you can do so in the code.""")
        
        exp1.dataframe(setup_df)
        #train the model
        #https://pycaret.gitbook.io/docs/get-started/functions/train
        try: 
            best_model = compare_models(budget_time=1, include=['lr', 'knn', 'nb', 'xgb', 'rf'], sort='Accuracy', turbo=True) 
            compare_df = pull()
            st.session_state.compare_df = compare_df
            st.session_state.best_model = best_model
            st.subheader("Bloody Oath that's an impressive table of ML models! :chart_with_upwards_trend:")
            st.write('The best model is at the top of the leaderboard. You can now review the model performance and make predictions.') 
            st.dataframe(compare_df) 
        except: 
            st.warning('There was an unexpected error - Please check the dataset and try again, or unfortunatly there could  other issues.')
            st.stop()
    
        #renders the best model leaderboard: 



        st.subheader('Best Model Trained:')
        st.info(best_model)
        st.session_state.best_model_desc= str(best_model)
        
        st.write('Time taken to train the model:', round(((time.time() - start_time)/60), 2), 'mins')
    
        st.image(width=200, image=f'https://gifdb.com/images/high/borat-very-nice-fist-pump-wq6v07qb55osz1jt.gif')
        st.subheader(':rainbow[Well done! You have trained your machine learning model, now you are ready to use it! Go to the navigation to continue.]:point_up:')



