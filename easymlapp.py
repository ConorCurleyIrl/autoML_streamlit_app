# 1. Importing libraries & multi-page setup
######################################################################y
import streamlit as st 
#st.set_page_config(layout="wide", page_title="Conor's EasyML App", page_icon=":rocket:", initial_sidebar_state="expanded")
from pycaret.classification import setup, compare_models, pull, save_model, load_model, predict_model, plot_model, evaluate_model, tune_model
import pandas as pd
import time
from streamlit_pandas_profiling import st_profile_report
from streamlit_option_menu import option_menu
from ydata_profiling import ProfileReport


#set the theme

# 2. Page configuration
######################################################################y

#layout

#cache the data & session state
@st.cache_resource(max_entries=10, ttl=3600)
def load_data():
    return None


#enable session state
if 'session_state' not in st.session_state:
    st.session_state.session_state = load_data()


# Load custom CSS for additional styling (if needed)
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")



# set up Sidebar
with st.sidebar:
    st.title("EasyML App Navigation:")
    st.image(width=100, image='https://i.pinimg.com/originals/cc/32/99/cc3299350f3d91327d4a8922ecae8fb8.gif', use_column_width=True)
    
    st.title("How to use this app:")
    st.write("Follow the steps in the sidebar menu above to build your own ML model. Here is a quick guide on the colour boxes & action buttons:")
    st.sidebar.info("This is an instruction box (blue), each step has instruction boxes to help you build your ML model.")
    st.sidebar.success('This is an sucess box (green), this should let you know when you have completed a step correctly.')
    st.sidebar.warning('This is an warning box (yellow), this is a warning to let you know something may be missing.')
    if st.button('This is an action button (red), Press it and shit happens :fire:',type='primary') == True:
            st.balloons()
            st.success('You rebel you :wink: You found the balloons button,  I think you are ready to start! :rocket:')
    
    st.subheader('App Created by Conor Curley')
    st.image(width=100,use_column_width=True,image=f'https://media.licdn.com/dms/image/v2/D4D03AQE1ykRQDFMyjA/profile-displayphoto-shrink_200_200/profile-displayphoto-shrink_200_200/0/1725466756415?e=1733356800&v=beta&t=tXoMk6tWslRQiMv5q2BTBtZS0gckSz3tYY9H6n0aetQ')
    st.write("Hope you enjoy the app :balloon: If you have any questions or feedback, please let me know via LinkedIn.")
    st.link_button('LinkedIn Link :wave:', 'https://www.linkedin.com/in/ccurleyds/')

    

st.session_state.selection2 = option_menu(None, ["Home", "Step1: Select Your Dataset", "Step2: Profile Your Dataset", 'Step3: Train Your Model','Step4: Review Model Performance','Step5: Test Future Predictions'], 
                                              icons=['house', 'cloud-upload', "clipboard-data", 'file-bar-graph-fill','rocket-takeoff','crosshair2'], 
                                              menu_icon="cast", default_index=0,styles={
                                                "container": {"padding": "0!important", "background-color": "#243664"},
                                                "icon": {"color": "orange", "font-size": "15px"}, 
                                                "nav-link": {"font-size": "12px", "text-align": "left", "margin":"0px", "--hover-color": "#262564"},
                                                "nav-link-selected": {"background-color": "green"},
                                                }, orientation="horizontal")

            


######################################################################y
# 3. Page Navigation
######################################################################y

if st.session_state.selection2 == "Home":
    import multi_pages._0_home as home
    home.homeapp()
elif st.session_state.selection2 == "Step1: Select Your Dataset":
    import multi_pages._1_select_data as select_data
    select_data.data_upload()
elif st.session_state.selection2 == "Step2: Profile Your Dataset":
    import multi_pages._2_profile_data as profile_data
    profile_data.profile()
elif st.session_state.selection2 == "Step3: Train Your Model":
    import multi_pages._3_model_build as model_build
    model_build.model()
elif st.session_state.selection2 == "Step4: Review Model Performance":
    import multi_pages._4_model_results as model_perf
    model_perf.results()
elif st.session_state.selection2 == "Step5: Test Future Predictions":
    import multi_pages._5_test_results as model_test
    model_test.test()


