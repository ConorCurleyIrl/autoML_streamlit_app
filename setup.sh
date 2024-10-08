mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"conor.francis.curley@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml



#https://medium.com/geekculture/deploy-your-streamlit-app-to-heroku-in-3-easy-steps-2804c4a3af58