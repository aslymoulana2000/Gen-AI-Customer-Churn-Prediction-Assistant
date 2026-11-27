import streamlit as st
from groq import Groq

#Groq API Connection Test - Configure the test application
st.set_page_config(
    page_title="Groq API Connection Test",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Groq API Connection Test")

st.write(
    "Testing the connection between our Python application "
    "and the GroqCloud LLM API."
)

#Load the API key securely
try:
    api_key = st.secrets["GROQ_API_KEY"]

except (KeyError, FileNotFoundError):
    st.error(
        "Groq API key not found. "
        "Please check your .streamlit/secrets.toml file."
    )

    st.stop()

#Initialize Groq client
try:
    client = Groq(api_key=api_key)

    st.info("Groq client initialized successfully.")

except Exception as error:
    st.error(f"Could not initialize Groq client: {error}")
    st.stop()

#Send a test request
try:
    with st.spinner("Connecting to GroqCloud..."):

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. "
                               "Give clear and concise answers."
                },
                {
                    "role": "user",
                    "content": "Explain customer churn in one sentence."
                }
            ],

            max_completion_tokens=200
        )

    #Extract the generated response
    ai_response = response.choices[0].message.content

    st.success("Groq API connection successful!")

    st.subheader("AI Assistant Response")

    if ai_response:
        st.write(ai_response)

    else:
        st.warning("The model returned an empty text response.")

except Exception as error:
    st.error("Groq API request failed.")
    st.write(str(error))

