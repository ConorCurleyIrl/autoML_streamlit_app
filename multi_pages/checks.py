    #configuring homepage
with st.sidebar:
    st.title("EasyML App Navigation:")
    st.image(width=100, image='https://i.pinimg.com/originals/cc/32/99/cc3299350f3d91327d4a8922ecae8fb8.gif', use_column_width=True)
    st.session_state.selection2 = option_menu(None, ["Home", "Step1: Select Your Dataset", "Step2: Profile Your Dataset", 'Step3: Train Your Model','Step4: Review Model Performance','Step5: Test Future Predictions'], 
                                              icons=['house', 'cloud-upload', "clipboard-data", 'file-bar-graph-fill','rocket-takeoff','crosshair2'], 
                                              menu_icon="cast", default_index=0,styles={
                                                "container": {"padding": "0!important", "background-color": "#243664"},
                                                "icon": {"color": "orange", "font-size": "15px"}, 
                                                "nav-link": {"font-size": "12px", "text-align": "left", "margin":"0px", "--hover-color": "#262564"},
                                                "nav-link-selected": {"background-color": "green"},
                                                }
                                              )
    st.session_state.selection2