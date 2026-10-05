import pandas as pd
import streamlit as st
from PIL import Image

def run():
    st.html('''
        <h1 style="text-align:center"> AG News Classification </h1>
        ''')
    image =  Image.open('news_hero.jpg')
    st.image(image)

    st.markdown('---')
    st.html('''
        <h2 style="text-align:center">Class distribution on train data</h2>
    ''')

    df_train = pd.read_csv('../train.csv', delimiter=',')
    plotted = df_train['Class Index'].value_counts().sort_index()

    plotted.index = ['World', 'Sports', 'Business', 'Sci/Tech']

    st.bar_chart(plotted, horizontal=True)

    st.markdown('---')
    st.subheader('About Project')
    st.html('''
        <p>This project aims to automatically input topic news to reduce time consumption if these task are done manually</p>
    ''')



if __name__ == '__main__':
    run()