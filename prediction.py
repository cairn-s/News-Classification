import re
import nltk
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt_tab')


def run():
    loaded_model = tf.keras.models.load_model('saved_model')
    st.html('''
        <h1>News Classification Prediction</h1>
        ''')

    with st.form('Predict news topic form'):
        news_inf = st.text_area(label='Type the news:', height=150)
        st.write('Avaiable topics: World, Sports, Business, Sci/Tech')
        submitted = st.form_submit_button('Predict')

    if submitted:
        if news_inf: 
        # Define Stopwords
        ## Load Stopwords from NLTK
            stop_words_en = stopwords.words("english")

            ## Create A New Stopwords
            new_stop_words = ['aye', 'mine', 'have']

            ## Merge Stopwords
            stop_words_en = stop_words_en + new_stop_words
            stop_words_en = list(set(stop_words_en))

            stemmer = PorterStemmer()

            def text_preprocessing(text):
                '''
                This function was used process texts and tokenization, e.g casefolding, remove mention, hashtags, whitespaces, URL, symbols

                Return: Processed words

                Example: 
                Before processing: "AP - Assets of the nation's retail money market mutual funds fell by  #36;1.17 billion in the latest week
                to  #36;849.98 trillion, the Investment Company Institute said Thursday."

                After processing: 'ap asset nation retail money market mutual fund fell billion latest week trillion invest compani institut said thursday'
                '''
                # Case folding
                text = text.lower()

                # Mention removal
                text = re.sub(r"@[A-Za-z0-9_]+", " ", text)

                # Hashtags removal
                text = re.sub(r"#[A-Za-z0-9_]+", " ", text)

                # Newline removal (\n)
                text = re.sub(r"\\n", " ",text)

                # Whitespace removal
                text = text.strip()
                text = re.sub(r"'s\b", "", text)

                # URL removal
                text = re.sub(r"http\S+", " ", text)
                text = re.sub(r"www.\S+", " ", text)

                # Non-letter removal (such as emoticon, symbol (like μ, $, 兀), etc
                text = re.sub(r"[^A-Za-z\s']", " ", text)

                # Tokenization
                tokens = word_tokenize(text)

                # Stopwords removal
                tokens = [word for word in tokens if word not in stop_words_en]

                # Stemming
                tokens = [stemmer.stem(word) for word in tokens]

                # Combining Tokens
                text = ' '.join(tokens)

                return text

            data_inf = pd.DataFrame([news_inf], columns=['text'])
            data_inf['text_processed'] = data_inf['text'].apply(lambda x: text_preprocessing(x))
            data_inf_np = np.array(data_inf['text_processed'])
            class_label = {
            0 : 'World',
            1 : 'Sports',
            2 : 'Business',
            3 : 'Sci/tech'
            }

            prediction = loaded_model.predict(data_inf_np)
            predicted_labels = np.argmax(prediction, axis=1)
            predicted_index = predicted_labels[0]
            st.write('**News topic:** ', class_label[predicted_index])
        else:
            st.error('Please type the news')
        

if __name__ == '__main__':
    run()