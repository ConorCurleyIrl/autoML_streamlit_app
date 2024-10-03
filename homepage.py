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
 
#cache the data
######################################################################
@st.cache_resource(max_entries=10, ttl=3600)
def load_data():
    return None

#enable session state
if 'session_state' not in st.session_state:
    st.session_state.session_state = load_data()


#configuring homepage
######################################################################

# Set the page configuration
st.set_page_config(layout="wide", page_title="Conor's EasyML App", page_icon=":rocket:", initial_sidebar_state="expanded")

# Load custom CSS for additional styling (if needed)
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

st.sidebar.subheader("Instructions:")
st.sidebar.write("This is a simple web app that allows you to build a Machine Learning model without writing code. Follow the steps in the sidebar menu to build your own ML model.")
st.sidebar.info("This is blue box is an instruction box. It will provide you with information on how to use the app. Follow the instructions in each step to build your ML model.")


# Homepage
######################################################################
st.title("Welcome to my EasyML App! :rocket:")
st.subheader("Build a Machine Learning model in minutes without writing code! :robot_face:")
st.image(width=400, image=f"https://lh4.googleusercontent.com/-yc4Fn6CZPtBPbRByD33NofqGnKGDrU5yy0t6ukwKKS5BxPLH5mUGLsetAUOtaK4D1oMp7otcLzuyr7khbRvCGvQjRSXJ5kjSbVOi3jbmHIjzHR7PO8mh52BlNgAHfnrViChn3jH5-z8M-A6M5OsK4c")

st.divider()

st.subheader("Who is this App for? 	:game_die:")
st.write("This app is for anyone who is interested in seeing the end-to-end Machine Learning process without writing code. It is designed to be simple and fun to use.")
st.write("This app has a simple interface for uploading your dataset, profiling your dataset and then running multiple Machine Learning algorithms to create a predictive model which you can test!")
st.write("I hope you enjoy using the app and building your own ML models!")

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

st.divider()

#easter egg 1
if st.button('DANGER :fire: This is a red action button, press it shit happens',type='primary') == True:
    st.balloons()
    st.success('You rebel you :wink: You found the balloons button,  I think you are ready to start! :rocket:')
    st.subheader(':rainbow[Click "Get Data (Step1)" in the sidebar to get this show on the road.] :point_left:')

st.divider()  

st.subheader('Created by Conor Curley')
st.image(width=180,image=f'https://media.licdn.com/dms/image/v2/D4D03AQE1ykRQDFMyjA/profile-displayphoto-shrink_200_200/profile-displayphoto-shrink_200_200/0/1725466756415?e=1733356800&v=beta&t=tXoMk6tWslRQiMv5q2BTBtZS0gckSz3tYY9H6n0aetQ')
st.write("Hope you enjoy the app :balloon: If you have any questions or feedback, please let me know via LinkedIn.")
st.link_button('LinkedIn Link :wave:', 'https://www.linkedin.com/in/ccurleyds/')
