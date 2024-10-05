######################################################################
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

######################################################################
# Upload data page
######################################################################

def profile():

    #cache the data
    @st.cache_resource(max_entries=10, ttl=3600)
    def load_data():
        return None

    #enable session state
    if 'session_state' not in st.session_state:
        st.session_state.session_state = load_data()

    ######################################################################
    #h_col3, Col2 = st.columns(2) 
        #Set up profile report
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Step 2::mag:", divider=True)
        st.subheader("Explore your Data to understand it better! :mag:")
        st.markdown("Instructions:")
        markdown = """1. Click the button down the page and have a gander at the report! :eyes:
                    """

        st.info(markdown)
    with col2:
        st.image(width=300, use_column_width=True, image=f'https://visme.co/blog/wp-content/uploads/2016/04/Header-1200-3.gif')
    
    
    st.divider()
    st.markdown("A sample of your dataset is displayed below:")
        # next steps prompt
    if not st.session_state.df.empty:  
        st.write('This datsaset has ' + str(st.session_state.df.shape[0]) + ' rows and ' + str(st.session_state.df.shape[1]) + ' columns.')
        st.dataframe(st.session_state.df.head(10))
    else:    
        st.warning('No dataset uploaded yet. Please upload a dataset to continue.')

    #generate the profile report
    if st.button('Make some pretty graphs for me', type='primary') == True:
        #create profile report
        start_time_pp = time.time()
        profile = ProfileReport(st.session_state.df, title='Profiling Report', explorative=True)
        
        #rendering the report in the streamlit app
        st.write('Your dataset profile report is being automatically created :tada:')
        st_profile_report(profile)
        st.write('Time taken to create data profile report', round(((time.time() - start_time_pp)/60), 2), 'mins')
    
        if profile is not None:
            st.success('Look at you go, you profiled your dataset! select "Step3" to continue :point_up_2:')
    
    
    
