# 1. importing libraries - see requirements.txt for all libraries used
#######################################################################
import streamlit as st 

from pycaret.classification import setup, compare_models, pull, save_model, load_model, predict_model, plot_model, evaluate_model, tune_model
import pandas as pd
import time
from streamlit_pandas_profiling import st_profile_report
from streamlit_option_menu import option_menu
from ydata_profiling import ProfileReport

#cache the data & session state
######################################################################


######################################################################
# Upload data page
######################################################################

def data_upload():

    #cache the data & session state
    @st.cache_resource(max_entries=10, ttl=3600)
    def load_data():
        return None
    #enable session state
    if 'session_state' not in st.session_state:
        st.session_state.session_state = load_data()

    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()

    #Homepage 
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Step 1: Select your dataset :file_folder:')
        st.write("Explore your Data to understand it better! :mag:")
        st.markdown("Instructions:")
        markdown = """1. Select a dataset to use or upload your own dataset. :file_folder:
                    """

        st.info(markdown)
        st.write('Note: this app can only perform solve classifIcation problems - predicting 1 or many outcomes, select a dataset that fits this requirement. ')

    with col2:
        st.image(width=300, use_column_width=True, image=f'https://c.tenor.com/eUsiEZP1DnMAAAAC/beam-me-up-scotty.gif')
  
    st.divider()

    st.subheader("Option 1: Select a Sample Dataset:")
    st.write("These datasets are open-source and can be used for educational purposes. I've included them in the app for you to use. I have slighyly modified the datasets to make them easier to use in the app - moved the order of coloumns, removed some columns and added some missing values. See App Build Notes for more info and links to original datasets.")

    col1, col2 = st.columns(2)
    
    with col1:
    # Add common datasets
        with st.container(border=True):
            #titanic dataset
            st.subheader('#1 Titanic Passenger Dataset :ship::')
            st.write("This is the Wonderwall of datasets - everyone who knows it, is sick of it but if you've never seen it then it's a banger!. It is a classic dataset used to train ML models, to predict if a passenger survived or not. using the passenger information.")
            st.image(width=300, image=f"https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FrvhpSE1rQFsnC%2F200.gif&f=1&nofb=1&ipt=61cb70717c6d7e13616619274bbbaf66e471d15d3751d767e03ad3060a91aeff&ipo=images")

            if st.button("Let's predict who survives", type='primary') == True:
                    st.session_state.df = pd.read_csv("data/titanic_data.csv", index_col=None)
                    st.success('Titanic dataset selected :ship: See a sample of the dataset below:')
                    st.dataframe(st.session_state.df.head(5))
            
    with col2:
        #telco company dataset
        with st.container(border=True):
            st.subheader('#2 Vodafone Customer Dataset: :phone:')
            st.write("This is the 'please don't leave me' dataset, used to predict when a customer leaves/churns. Before you ask, yes Churn is a silly business term invented to sound technical. Also this is not actually Vodafone data, it's a sample dataset from a telecommunications company (telco) but helps the learning process to think of a similar company.")
            st.image(width=300, image=f"https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fc.tenor.com%2F8FKTJDMvH2IAAAAC%2Fhomer-simpsons.gif&f=1&nofb=1&ipt=0dcde120abcd6aa2f5a6520116b46ccbeeef91232629975183ddbbbda791cb2f&ipo=images")

            if st.button("Let's predict customer churn", type='primary') == True:
                st.session_state.df = pd.read_csv("data/telco_churn.csv", index_col=None)
                st.success('Vodafone dataset selected :phone: See a sample of the dataset below:')
                st.dataframe(st.session_state.df.head(5))

    col3, col4 = st.columns(2)
    
    with col3:
        # Penguins dataset
        with st.container(border=True):
            st.subheader('#3 Penguins Species Classification Dataset :penguin:')
            st.write("This is the 'Which penguin can I steal?' dataset, used to predict the species of penguins based on some observations. There are 3 different species of penguins in this dataset.")
            st.image(width=300, image=f"https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fc.tenor.com%2FOOslTDt3rHAAAAAd%2Fpenguin-of-madagascar.gif&f=1&nofb=1&ipt=079f6fc56a0fb949a4ff5bc2d4607e830fc8d179ed7389aa34fc56703249409d&ipo=images")

            if st.button("Which furball is coming home?", type='primary') == True:
                st.session_state.df = pd.read_csv("data/penguins.csv", index_col=None)
                st.success('Penguin dataset selected :penguin: See a sample of the dataset below:')
                st.dataframe(st.session_state.df.head(5))
    with col4:
        # Mushroom dataset
        with st.container(border=True):
            st.subheader('#4 Mushroom Classification Dataset :mushroom:')
            st.write("This is the 'Should Mario eat this?' dataset, used to predict if a mushroom is edible or poisonous based on some observations. Also this has +60,000 rows so it's a big one!")
            st.image(width=300, image=f"https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fi.pinimg.com%2Foriginals%2F4d%2F4c%2Ffc%2F4d4cfc0fa82e58789f811bda40414bc0.gif&f=1&nofb=1&ipt=d338a8039bb3d70ae5d8198661e3f7da03bae8417b9b4cae095e11841301b9c5&ipo=images")

            if st.button("Which mushrooms would kill Mario?", type='primary') == True:
                st.session_state.df = pd.read_csv("data/mushroom_dataset.csv", index_col=None)
                st.success('Mushroom dataset selected :mushroom: See a sample of the dataset below:')
                st.dataframe(st.session_state.df.head(5))

    st.divider()
    st.subheader("Option 2: Upload your own dataset:")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")  

    if uploaded_file :
        # Read the uploaded file:
        st.session_state.df = pd.read_csv(uploaded_file,index_col=None)   
        st.session_state.df_holdout= st.session_state.df[:-10]               

        # Display the dataset:
        st.success('Dataset uploaded successfully! Here is a sample of your dataset:')
        st.dataframe(st.session_state.df.head(5))

    # next steps prompt
    st.divider()

    if not st.session_state.df.empty:  
        st.success('A Dataset has been selected, ready to move to the next step!')
        st.dataframe(st.session_state.df.head(5))
        st.subheader(':rainbow[Great job you have a dataset loaded! Select "Step2" in the navigation to continue.] :point_up_2:')
        
    else:    
        st.warning('No dataset uploaded yet. Please upload a dataset to continue.')
        st.subheader(':rainbow[Select a sample dataset or upload your own dataset to continue.] :point_up_2:')
    #configuring homepage

