
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



#############################################################################################
## App Build Notes
#############################################################################################


st.subheader('App Build Notes')
st.image(width=300, image=f'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia.tenor.com%2FUg6cbVA1ZsMAAAAM%2Fdeveloper.gif&f=1&nofb=1&ipt=7285c5bfb06e6eae122b6d0a5d10980b726559be7da57a57848a303739a2738e&ipo=images')
st.subheader('How I built this app:')
st.info('I built this app using the Streamlit framework for the web interface and the Pycaret library for the machine learning models.')

exp2 =st.expander("Some Notes & Learnings: (9-June-2024)")
exp2.st.markdown("- Well its now live and I'm pretty happy with how it turned out. Learned a hell of alot about streamlit, pycaret and more python in general")
exp2.st.markdown("- Few issues getting the package management right, 132 packages currently are needed. This app is now on Python 3.10 (using other versions created package conflicts) and I added the absolute minimum list of libraries in the requirements.txt so I could to reduce errors then use Pycaret and scikit-learn which have a lot of included dependencies so once you have them installed, you are good to go I've learned after many hours debugging. I also kept some libraries not pinned to a specific version and let the streamlit package manager avoid conflicts.")
exp2.st.markdown("- Streamlit is a great tool for building simple web app with Python but it has some limitations - an example is that there is no multi-page functionality. I had to use a lot of workarounds (navigation menu) to get the app to work as I wanted but now I've the hnag of it so I think I can build more complex apps in the future.")
exp2.st.markdown("- Userflow could be a bit better - trying to strike a balance between simplicity and showcasing the ML functionality was tricky. Ideally you would have full ML model training settings available in a sidebar to allow full customisation but I wanted to keep it simple for new users. I may add this in the future.")
exp2.st.markdown("- Next time, I would plan out the app UX structure better next time - I had to refactor a lot of code as I went along to make it more user-friendly.")
exp2.st.markdown("- Hosting the app on streamlit's public infastructure was tricky with the ML modelling need for compute. I had to configure and simplify the modelling process, previous ML training took 24mins! to work with the free tier limitations. But hey its free so I can't complain too much. Let's see how it goes with more users.")
exp2.st.markdown("- I may deploy on Heroku next time for hosting as it has more flexibility and better performance. ")
exp2.st.markdown("- All in all, I learned a lot building this app and I'm excited to build and release more ML apps.")
exp2.st.markdown("- Thanks for checking out the app! If you have any feedback or questions, please let me know via LinkedIn.")

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
