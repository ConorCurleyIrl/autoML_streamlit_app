import pickle
model = pickle.load(open('c:/Users/Conor/autoML_streamlit_app/best_model.pkl', 'rb'))
model


    

    if uploaded_model is not None:
        up_model = load_model(uploaded_model)
        st.success("Model uploaded")
        st.write(up_model)

#############################################################################################
## ML Glossary Section
#############################################################################################

if choice == 'ML Glossary':
    st.title('Machine Learning Glossary:')
    st.image('https://i.pinimg.com/originals/cc/32/99/cc3299350f3d91327d4a8922ecae8fb8.gif')
    
    st.subheader('Technology I used to build this app:')
    st.link('Streamlit', 'https://www.streamlit.io/')
    st.link('Pycaret', 'https://pycaret.org/')
    st.link('Ydata Profiling', 'https://pypi.org/project/ydata-profiling/')
            
    st.header('Explain Classification Problems:')
    st.info(short_class_desc)

    st.subheader('Explain AutoML:')
    st.info(short_automl_desc)

    st.subheader('Explain Data Profiling')
    st.info(short_profile_desc)