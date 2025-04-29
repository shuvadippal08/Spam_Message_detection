import streamlit as st
import pickle
import string
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers and special characters
    text = re.sub(r'\d+', '', text)
    #Tokenization
    text = nltk.word_tokenize(text)
    #changing sentence form
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    #remove stopwords
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    # Optionally
    return text


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")
if st.button('Predict'):

    # 1. preprocess
    transformed_sms = preprocess_text(input_sms)
    final_input = " ".join(transformed_sms)
    # 2. vectorize
    vector_input = tfidf.transform([final_input])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")