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
    st.write("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            st.write(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    st.write("...all files ready")
    st.write()

def initialize_model():
    """Initializes the Generative Model."""
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
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
        <div style="display: flex; align-items: center;">
            <img src="https://www.bacancytechnology.com/blog/wp-content/uploads/2019/08/ezgif.com-optimize-5.gif" width="90" height="90" style="margin-right: 20px;">
            <h1>Dataset Analysis with Sarah</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar for file upload
    st.sidebar.header("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        with open("uploaded_file.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file = upload_to_gemini("uploaded_file.csv", mime_type="text/csv")
        wait_for_files_active([file])
        
        # Initialize Model
        model = initialize_model()
        
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
        
        # Profile Report
        df = pd.read_csv("uploaded_file.csv")
        profile = ProfileReport(df)
        
        with st.expander("Profile Report"):
            st_profile_report(profile)
        
        # Chat history and user interaction
        st.header("Chat with Sarah")
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        for chat in st.session_state.chat_history:
            if chat.startswith("User:"):
                st.text_area("You:", value=chat[len("User:"):].strip(), height=100, key=chat, disabled=True)
            elif chat.startswith("Model:"):
                st.text_area("Sarah:", value=chat[len("Model:"):].strip(), height=100, key=chat, disabled=True)
        
        # User Query Section
        user_query_col, submit_button_col = st.columns([4, 1])
        with user_query_col:
            user_query = st.text_input("Ask Sarah", key="user_query")
        
        with submit_button_col:
            st.markdown(
                """
                <style>
                .submit-container {
                    margin-top: 0px;  /* Adjust this value to align perfectly */
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            with st.container():
                st.markdown('<div class="submit-container">', unsafe_allow_html=True)
                submit_button = st.button("Submit", key="submit_button")
                st.markdown('</div>', unsafe_allow_html=True)
        
        if submit_button and user_query:
            # Add user query to chat history
            st.session_state.chat_history.append(f"User: {user_query}")
            
            # Start conversation with model
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
            
            # Add model response to chat history
            st.session_state.chat_history.append(f"Model: {response.text}")

            # Display model response
            st.text_area("Sarah:", value=response.text, height=100, key=response.text, disabled=True)
            
            # Get response accuracy
            accuracy = get_response_accuracy(response.text, model)
            st.write(f"Model-assessed accuracy: {accuracy}")

            # Save chat history to a file (optional)
            # with open(f"{file.display_name}_chat.txt", "a") as chat_file:
            #    chat_file.write(f"User: {user_query}\nModel: {response.text}\nAccuracy: {accuracy}\n\n")

    # Footer section
    st.markdown("""
        <style>
        .footer {
            position: fixed;
            bottom: 10px;
            right: 10px;
            background-color: rgba(0, 0, 0, 0); /* Transparent background */
            text-align: center;
            padding: 5px 10px;
            border-radius: 5px;
        }
        </style>
        <div class="footer">
            <p>Developed by </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Directly set your Google API key here
    api_key = "AIzaSyDc00q-vSfqjBb_Rjf292g7SjfPi6Iu3m8"
    genai.configure(api_key=api_key)
    main()
