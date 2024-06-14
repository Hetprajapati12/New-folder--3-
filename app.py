import streamlit as st

from rag_model import get_response
st.title("PDF-based Q&A Chatbot")

question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    answer = get_response(question)
    st.write("Answer:", answer)
