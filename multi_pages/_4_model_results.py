
def results():
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


    #cache the data
    @st.cache_resource(max_entries=10, ttl=3600)
    def load_data():
        return None

    #enable session state
    if 'session_state' not in st.session_state:
        st.session_state.session_state = load_data()

    import seaborn as sns
    import matplotlib.pyplot as plt


    import streamlit as st
    import matplotlib.pyplot as plt


    def load_data():
        return None

    # Enable session state
    if 'session_state' not in st.session_state:
        st.session_state.session_state = load_data()

    st.subheader('Step 4: Review the model performance statistics and graphs below. (Optional) ')

    st.divider()

    # Check if best_model is loaded
    if 'best_model' in st.session_state and st.session_state.best_model is not None:
        st.subheader('Best Model Trained:')
        st.info(st.session_state.best_model_desc)

        # Load the best model
        st.subheader('Model Performance Table:')
        st.dataframe(st.session_state.compare_df)


        # Map short model names to full names
        model_name_mapping = {
            'lr': 'Logistic Regression',
            'knn': 'K-Nearest Neighbors',
            'nb': 'Naive Bayes',
            'ridge': 'Ridge Classifier',
            'rf': 'Random Forest',
            'gbc': 'Gradient Boosting Classifier',
        }
        # Replace the index with full model names
        st.session_state.compare_df.index = st.session_state.compare_df.index.map(model_name_mapping)
        # Convert accuracy to percentage
        st.session_state.compare_df['Accuracy_perc'] = st.session_state.compare_df['Accuracy'] * 100
        
        # Create a bar plot using Seaborn
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(x=st.session_state.compare_df['Accuracy_perc'], y=st.session_state.compare_df['Model'], data=st.session_state.compare_df, palette='viridis')
        # Add data labels inside the bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f%%', label_type='center', color='white', fontsize=10)
        # Add gridlines
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        # Enhance aesthetics
        plt.title('Top Models Compared by Prediction Accuracy (%)', fontsize=16, fontweight='bold')
        plt.xlabel('Accuracy (%)', fontsize=12)
        plt.ylabel('Model Name', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        st.pyplot(plt)
        


        # Fig 1 AUC graph
        plt.figure(figsize=(8, 5))
        st.subheader('Fig 1 Model performance: AUC curve')
        st.write('AUC graph is very useful when the target variable is binary. It is a measure of how well a binary classification model is able to distinguish between positive and negative classes.')
        try:
            plot_model(st.session_state.best_model, plot="auc", display_format="streamlit")
        except TypeError as e:
            st.error(f"Error plotting AUC curve: {e}")

        # Fig 2 Confusion Matrix
        plt.figure(figsize=(8, 5))
        st.subheader('Fig 2 Model performance: Confusion Matrix')
        st.write('The confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known.')
        try:
            plot_model(st.session_state.best_model, plot="confusion_matrix", display_format="streamlit")
        except TypeError as e:
            st.error(f"Error plotting Confusion Matrix: {e}")

        # Fig 3 Precision-Recall Curve
        plt.figure(figsize=(8, 5))
        st.subheader('Fig 3 Model performance: Precision-Recall Curve')
        st.write('The precision-recall curve shows the trade-off between precision and recall for different threshold settings.')
        try:
            plot_model(st.session_state.best_model, plot="pr", display_format="streamlit")
        except TypeError as e:
            st.error(f"Error plotting Precision-Recall Curve: {e}")

        # Fig 4 Feature Importance
        plt.figure(figsize=(8, 5))
        st.subheader('Fig 4 Model performance: Feature Importance')
        st.write('Feature importance refers to techniques that assign a score to input features based on how useful they are at predicting a target variable.')
        try:
            plot_model(st.session_state.best_model, plot="feature", display_format="streamlit")
        except TypeError as e:
            st.error(f"Error plotting Feature Importance: {e}")

        st.success("Well done, that was a stats heavy section :brain: You are ready to test your model's prediction power. Go to the navigation to continue.")
        # Performance Scores
        st.subheader("Performance Scores:")

        expander = st.expander("Optional: So what do these performance scores mean? Click here for more info")
        expander.subheader("Performance Scores:")   

        expander.subheader("1. What does 'Accuracy' mean?")
        expander.write("Accuracy is the proportion of true results (both true positives and true negatives) among the total number of cases examined.")

        expander.subheader("2. What does 'Precision' mean?")
        expander.write("Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.")

        expander.subheader("3. What does 'Recall' mean?")
        expander.write("Recall is the ratio of correctly predicted positive observations to the all observations in actual class.")

        expander.subheader("4. What does 'F1' mean?")
        expander.write("F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account.")

        expander.subheader("5. What does 'Kappa' mean?")
        expander.write("Kappa is a classification accuracy normalized by the imbalance of the classes in the data.")

        expander.subheader("6. What does 'MCC' mean?")
        expander.write("Matthews correlation coefficient (MCC) is used in machine learning as a measure of the quality of binary classifications.")

        expander.subheader("7. What does 'AUC' mean?")
        expander.write("AUC stands for Area Under the ROC Curve. This metric is used to evaluate the performance of a classification model.")

        expander.subheader("8. What does 'Log Loss' mean?")
        expander.write("Logarithmic loss (Log Loss) measures the performance of a classification model where the prediction output is a probability value between 0 and 1.")

        expander.subheader("9. What does 'Time taken' mean?")
        expander.write("Time taken to train the model.")

        st.subheader(':rainbow[Good job, you have reviewed your model! Go to the navigation to continue.]:point_left:')

    else:
        st.warning("A trained model was not found. Please ensure the model is trained, perhaps repeat step3?.")
