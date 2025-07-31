import streamlit as st
import os
from dotenv import load_dotenv
import langchain_google_genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

## Langsmith Tracking (optional)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Question and Answer Chatbot"

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries."),
        ("user", "Question: {question}"),
    ]
)

# Function to generate response from Gemini
def generate_response(question, api_key, llm_name, temperature, max_tokens):
    os.environ["GOOGLE_API_KEY"] = api_key  # Required by `google-generativeai`
    
    llm = ChatGoogleGenerativeAI(
        model=llm_name,
        temperature=temperature,
        max_output_tokens=max_tokens
    )

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Streamlit UI
st.title("🤖 Gemini-Powered Q&A Chatbot")

st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")

# Gemini model options
llm_name = st.sidebar.selectbox("Select a Gemini Model", [
    "gemini-1.5-flash", 
    "gemini-1.5-flash-latest",  
])

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=1024, value=256)

st.write("Ask me anything!")
user_input = st.text_input("You:")

if user_input:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    else:
        try:
            response = generate_response(user_input, api_key, llm_name, temperature, max_tokens)
            st.markdown(f"Assistant: {response}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.write("Enter a question above to get started!")

