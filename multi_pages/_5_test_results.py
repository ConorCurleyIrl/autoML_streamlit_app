def test():
    #######################################################################
    #importing libraries - see requirements.txt for all libraries used
    ######################################################################y
    import streamlit as st
    from pycaret.classification import setup, compare_models, pull, save_model, load_model, predict_model, plot_model, evaluate_model, tune_model
    import pandas as pd
    import time
    from streamlit_pandas_profiling import st_profile_report

    # ydata_profiling is the latest howerver it is not supported by streamlit_pandas_profiling so using older versio pandas_profiling
    #from pandas_profiling.profile_report import ProfileReport
    from ydata_profiling import ProfileReport
    import seaborn as sns
    import matplotlib.pyplot as plt


    import os #file managementstreamlit multiple pages 2024MMM
    import gc #garbage collection


    st.subheader('Step4: Predict the Future! :rocket:')
    st.image(width=400, image='https://media1.giphy.com/media/ZhESFK96NxbuO1yDgy/giphy.gif')
    st.subheader('Instructions:')
    st.info("Click the button below to run the model on new data and get some predictions.")

    st.divider()

    st.subheader("Let's review, What we have done so far:")
    st.write("1. We have loaded the data and cleaned it.")
    st.write("2. We have trained a machine learning model on the data.")
    st.write("3. We have reviewed the model performance statistics and graphs.")
    st.write("4. Now we will test the model on new data and get some predictions.")

    st.divider()

    st.subheader("So lets test our model on some new data! :rocket:")
    st.write("when the model was being trained, we held out some data (5%) to showcase the model later on. We will now use this data to test the model.")
    st.write("This is our model")
    st.write(st.session_state.best_model)

    st.write("This is how unseen data looks like: the target variable is not included in the data as it is what we are trying to predict.")
    st.dataframe(st.session_state.df_holdoutX)

    st.divider()

    if st.button("Alrighty this is the moment of truth - let's punch those numbers and predict with our model!", type='primary') == True:

        st.write('Applying our ML model ...')
        new_prediction = predict_model(st.session_state.best_model, data=st.session_state.df_holdoutX)
        new_prediction['Target_Variable_Actual'] = st.session_state.df_holdouty
        st.success('Predictions made successfully! Have a look at the predictions above in the table! (at the end you will see your prediction and actual classification):rocket:')    
        new_prediction['correct_prediction'] = new_prediction['prediction_label'] == new_prediction['Target_Variable_Actual']
        st.dataframe(new_prediction)
        
        #create a barchart of the predictions
        new_prediction['correct_prediction'] = new_prediction['prediction_label'] == new_prediction['Target_Variable_Actual']
        
        st.subheader("Predictions made by the model:")
        accuracy_score = new_prediction['correct_prediction'].sum() /new_prediction['correct_prediction'].count()
        holdout_df_size= new_prediction['Target_Variable_Actual'].count()
        st.write(f'Accuracy of the model is: {accuracy_score*100}% from our holdout dataset, {holdout_df_size} observations were used.')
        st.write('The barchart below shows the number of correct and incorrect predictions made by the model.')
        st.bar_chart(new_prediction['correct_prediction'].value_counts())
        st.success('Predictions made successfully!:rocket: But can we do better? Go back and try tuning the model to improve the accuracy by ignoring some columns.')
       
    st.divider()

    #easter egg
    st.success("Gotten this far? I think you deserve a dance!:")	
    if st.button("Dance button!", type='primary') == True:
        st.image(width=500,image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fc.tenor.com%2FZAoUo4PquusAAAAC%2Fyou-did-it-congratulations.gif&f=1&nofb=1&ipt=bfeb6b6934c23145f401edd23610973857097a7938a18d49835bc9dbbc30e0f1&ipo=images')
        st.subheader(":rainbow[Congrats you beauty! You built your own Machine learning model without writing a single line of code! ]:wink:")
        st.balloons()
        st.balloons()
        st.balloons()
        
