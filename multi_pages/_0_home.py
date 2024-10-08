# Homepage Function
######################################################################

def homeapp():

    #import libraries
    import streamlit as st 
    from pycaret.classification import setup, compare_models, pull, save_model, load_model, predict_model, plot_model, evaluate_model, tune_model
    import pandas as pd
    import time
    from streamlit_pandas_profiling import st_profile_report
    from streamlit_option_menu import option_menu
    from ydata_profiling import ProfileReport

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
    col1, col2  = st.columns(2)

    with col1:
        st.title("Conor's EasyML App:rocket:")
        st.subheader("Build a Machine Learning model in minutes without writing code! :robot_face:", divider="gray")
        st.subheader("But who is this App for? 	:game_die:")
        st.write("This app is for anyone who is interested in seeing the end-to-end Machine Learning process without writing code. It is designed to be simple and fun to use.")
        st.write("This app has a simple interface for uploading your dataset, profiling your dataset and then running multiple Machine Learning algorithms to create a predictive model which you can test!")
        st.write("I hope you enjoy using the app and building your own ML models!")
    with col2: 
        st.image(width=300,use_column_width=True, image="https://lh4.googleusercontent.com/-yc4Fn6CZPtBPbRByD33NofqGnKGDrU5yy0t6ukwKKS5BxPLH5mUGLsetAUOtaK4D1oMp7otcLzuyr7khbRvCGvQjRSXJ5kjSbVOi3jbmHIjzHR7PO8mh52BlNgAHfnrViChn3jH5-z8M-A6M5OsK4c")

    

    st.divider()
    st.subheader("A little more info: :mag_right:")
    expander = st.expander("Why did I build this? :building_construction:")
    expander.write(""" People are often put off Machine Learning due to the complexity but I've always thought of it as a fun puzzle game! Hopefully this app can take a little of the mystery and seriousness out of it!.""")
    expander.write(""" Also, you only learn by doing and I wanted to learn more about the Streamlit framework for simple web development and the Pycaret AutomMl package as a prototyping tool""")

    expander1 = st.expander("Ok, what is Machine Learning (ML)? What is ML model? :robot_face:")
    expander1.write("Machine learning is a branch of artificial intelligence that uses computer algorithms to learn from data and perform tasks that normally require human intelligence.")
    expander1.info("An ML model is a set of complex rules that uses historical and current data to predict future outcomes")
    expander1 = st.expander("What is AutoML and why is it useful? :computer:")
    expander1.write("""
            AutoML is a process of automating the end-to-end process of applying Machine Learning to real-world problems.
            This app is designed to make the process of building ML models easier and faster.
                    
            Building production Machine Learning models requires a lot of time, high quality data, platform infrastructure, effort, and expertise. 
            But with the advent of AutoML, the process has become much easier and faster to build basic starter models.
            
            """)
    expander1.info("This app used AutoML technology to build your own ML model without writing code.")

    expander2= st.expander('Will my model be as good as one built by an experienced Data Scientist? :microscope:') 
    expander2.write("Well no, but they will be pretty good and will be a great starting point for understanding your data and making initial predictions.")

    expander2 = st.expander('Ok so how do I use this app? :racing_car:')
    expander2.write("Follow the steps in the navigation menu and in a few clicks you'll have a Machine Learning model trained on historical data that can provide future predictions.")

    st.divider()
