import streamlit as st
import eda
import prediction

st.set_page_config(page_title='News Classification',
                   layout='wide',
                   initial_sidebar_state='collapsed')

page = st.sidebar.selectbox('Choose page', ('EDA', 'Prediction Page'))

if page == 'EDA':
    eda.run()
else:
    prediction.run()