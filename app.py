import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai
import numpy as np

# Configure the API key for Google Generative AI
gemini_api_key = "AIzaSyBX8W4DGmHrwJQsHNe0uMn7rhD704SLZBg"
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-pro')

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text().lower()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def get_keywords_from_text(data):
    try:
        prompt = f"Can you generate 10 key words according to this paragraph, comma-separated string? If you can't, just respond 'no, sorry'.\n{data}"
        response = model.generate_content(prompt)
        keywords = response.text.split(", ")
        return keywords
    except Exception as e:
        st.error(f"Error generating keywords: {e}")
        return ["no, sorry"]

def process_pdf_text(text):
    try:
        cluster_paragraphs = text.split("\n\n")  # Example of splitting paragraphs
        chat_data = [{"data": data, "keywords": get_keywords_from_text(data)} for data in cluster_paragraphs]
        return chat_data
    except Exception as e:
        st.error(f"Error processing text: {e}")
        return []

def get_relevant_data(chat_data, user_question):
    try:
        list_of_words = user_question.split()
        score_list = [sum(1 for y in x["keywords"] if y in list_of_words) for x in chat_data]
        indices = np.argsort(score_list)[-3:]  # Get top 3 matches
        ref_data = [chat_data[i]["data"] for i in indices]
        return ref_data
    except Exception as e:
        st.error(f"Error getting relevant data: {e}")
        return []

def get_answer(ref_data, user_question):
    try:
        prompt = f"Hello, I have some paragraphs you need to understand and answer the question.\n\n{str(ref_data)}\n\nMy question: {user_question}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer."

def main():
    st.title("PDF Question Answering App")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        if text:
            chat_data = process_pdf_text(text)
            st.write("PDF processed successfully. You can now ask questions about its content.")

            user_question = st.text_input("Ask a question about the PDF content:")
            if st.button("Get Answer") and user_question:
                ref_data = get_relevant_data(chat_data, user_question)
                answer = get_answer(ref_data, user_question)
                st.write("Answer:", answer)

if __name__ == "__main__":
    main()
