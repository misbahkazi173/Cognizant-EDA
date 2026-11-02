import os
import time
import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
import google.generativeai as genai
from streamlit_pandas_profiling import st_profile_report

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    st.sidebar.write(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    """Waits for the given files to be active."""
    st.sidebar.write("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            st.write(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    st.sidebar.write("...all files ready")
    st.sidebar.write()

def initialize_model():
    """Initializes the Generative Model."""
    generation_config = {
        "temperature": 0.5,  # Adjusted for more creative responses
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 2097152,
        "response_mime_type": "text/plain",
    }

    return genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
    )

def get_dataset_summary(file, model):
    """Get a quick summary of the dataset."""
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    file,
                    "whats the content ",
                ],
            },
            {
                "role": "model",
                "parts": [
                    "",
                ],
            },
        ]
    )
    response = chat_session.send_message("Give a quick summary of the dataset in a paragraph")
    return response.text

def get_suggestions(file, model):
    """Get suggestions for the most likely questions."""
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    file,
                    "What are the 10 most likely questions a user may ask about this dataset?",
                ],
            },
            {
                "role": "model",
                "parts": [
                    "",
                ],
            },
        ]
    )
    response = chat_session.send_message("Give 10 most likely questions")
    return response.text.split('\n')

def get_response_accuracy(response_text, model):
    """Get the model's self-assessment of the response accuracy."""
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    response_text,
                    "Rate the accuracy of this response from 1 to 5",
                ],
            },
            {
                "role": "model",
                "parts": [
                    "",
                ],
            },
        ]
    )
    response = chat_session.send_message("Rate the accuracy of the response")
    return response.text.strip()

def main():
    st.set_page_config(layout="wide")  # Set wide layout for better spacing

    # Title with image
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: center; text-align: center;">
            <img src="https://www.bacancytechnology.com/blog/wp-content/uploads/2019/08/ezgif.com-optimize-5.gif" width="100" height="100" style="margin-right: 20px;">
            <h1>Intelligent Dataset Analysis </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    #st.sidebar.image("https://www.drupal.org/files/cognizant%20logo.jpg", width=200)

    # File Upload in Sidebar
    st.sidebar.header("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        with open("uploaded_file.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file = upload_to_gemini("uploaded_file.csv", mime_type="text/csv")
        wait_for_files_active([file])
        
        # Initialize Model
        model = initialize_model()
        ##
                # Main content area
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Dataset Summary
            st.header("Dataset Summary")
            summary = get_dataset_summary(file, model)
            st.write(summary)
        
        with col2:
            # Likely Questions
            st.header("Likely Questions")
            suggestions = get_suggestions(file, model)
            for i, question in enumerate(suggestions, start=1):
                st.write(f"{question}")

        
        # Load the dataset
        df = pd.read_csv("uploaded_file.csv")
        profile = ProfileReport(df)
        
        # Display Profile Report in a dropdown
        with st.expander("Profile Report"):
            st_profile_report(profile)
        
        
        # Initialize chat history and accuracy ratings
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        st.header("Chat History")
        for chat in st.session_state.chat_history:
            st.write(chat)

        # User Query Section
        #st.header("Ask Your Query (Including Subjective Questions)")
        st.markdown(
            """
            <div style="display: flex; align-items: center;">
                <h2>Ask Your Query</h2>
                <img src="https://media.tenor.com/2YQj1-QHrtMAAAAi/vutura-chatbot.gif" width="120" height="120" style="margin-right: 20px;">
            </div>
            """,
            unsafe_allow_html=True
        )



        # Fix the user input text box at the footer
        query_container = st.container()
        with query_container:
            
            user_query = st.text_input("Enter your query", key="user_query", on_change=None)
            

        if user_query:
            chat_session = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [
                            file,
                            user_query,
                        ],
                    },
                    {
                        "role": "model",
                        "parts": [
                            "",
                        ],
                    },
                ]
            )
            response = chat_session.send_message(user_query)
            st.write(response.text)
            
            # Get response accuracy
            accuracy = get_response_accuracy(response.text, model)
            st.write(f"Model-assessed accuracy: {accuracy}")

            # Update chat history
            st.session_state.chat_history.append(f"User: {user_query}")
            st.session_state.chat_history.append(f"Model: {response.text} (Accuracy: {accuracy})")

            # Save chat history to a file
            with open(f"{file.display_name}_chat.txt", "a") as chat_file:
                chat_file.write(f"User: {user_query}\nModel: {response.text}\nAccuracy: {accuracy}\n\n")

if __name__ == "__main__":
    # Directly set your Google API key here
    api_key = "AIzaSyC8jYNEOQEnjGZruorG9qpuc8GO79DNyBc"
    genai.configure(api_key=api_key)
    main()
