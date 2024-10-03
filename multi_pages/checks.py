'''


# Create the sidebar navigation
st.sidebar.title("Navigation")

with st.navigation("Sidebar Menu"):
    
    selection = option_menu("Main Menu", ["Home", 'Settings'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)
    selection
'''