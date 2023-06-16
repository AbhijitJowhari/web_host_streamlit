import streamlit as st
import nltk
nltk.download("punkt")

#giving a title
st.title("Post your Doubts")


#getting the question from the user
name=st.text_input("Your Name")
Roll_no=st.text_input("Your IITK Roll No.")
question=st.text_input("Ask Your Question")


#code for prediction
category=''

#creating a button for predicting the category
from prediction import predict

if st.button('Question Category'):
    category=predict(question)
    
st.success(category)
