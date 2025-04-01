import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

# Streamlit app configuration
st.set_page_config(
    page_title="Mental Health Chatbot",
    layout="wide",
    )

st.markdown("""
    <style>
        .stApp {
            background-color: #F5F7FA;
            font-family: 'Arial', sans-serif;
        }
        .stChatMessage {
            border-radius: 12px;
            padding: 14px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stChatMessage.user {
            background-color: #E3F2FD;
            color: #0D47A1;
            border-left: 4px solid #42A5F5;
        }
        .stChatMessage.assistant {
            background-color: #E8F5E9;
            color: #1B5E20;
            border-left: 4px solid #66BB6A;
        }
        .stTextInput>div>div>input {
            background-color: #FFFFFF;
            border-radius: 8px;
            padding: 10px;
        }
        h1 {
            color: #5E35B1;
            text-align: center;
        }
        .stMarkdown {
            color: #455A64;
        }
        .stButton>button {
            background: linear-gradient(90deg, #3F51B5 0%, #43A047 100%);
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
        }
        .tab-content {
            padding: 10px;
            border-radius: 8px;
            background-color: #FFFFFF;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        st.error(f"Vector store loading error: {str(e)}")
        return None


def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
        max_new_tokens=512,
        return_full_text=False
    )


def get_qa_chain(llm, vectorstore):
    CUSTOM_PROMPT_TEMPLATE = """[INST]
    Use only the provided context to answer the question.
    If you don't know, say "I don't know". 
    Be concise and factual.

    Context: {context}
    Question: {question}

    Answer directly: [/INST]"""

    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )


def main():
    # App header
    st.title("Mental Health Assistant")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Chat input
    if prompt := st.chat_input("I help you understand Mental Health..."):
        # Add user message to chat history
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Process query
        try:
            HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
            HF_TOKEN = os.getenv("HF_TOKEN")

            vectorstore = get_vectorstore()
            if vectorstore:
                llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
                qa_chain = get_qa_chain(llm, vectorstore)

                response = qa_chain.invoke({'query': prompt})
                result = response["result"]

                # Display assistant response
                st.chat_message('assistant').markdown(result)
                st.session_state.messages.append(
                    {'role': 'assistant', 'content': result}
                )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
