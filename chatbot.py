import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
DB_FAISS_PATH = "vectorstore/db_faiss"

# Custom CSS for sidebar, main area, sample questions, and clear chat button
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #e6f0fa !important; /* Light blue tinge */
            font-family: 'Arial', sans-serif;
        }
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #87ceeb !important; /* Sky blue */
        }
        /* Sidebar text */
        [data-testid="stSidebar"] .css-1v0mbdj,
        [data-testid="stSidebar"] .css-1d391kg,
        [data-testid="stSidebar"] .css-1cypcdb {
            color: #000 !important;
        }
        /* Remove box around sample questions */
        .sample-question-btn {
            background: none !important;
            border: none !important;
            color: #000 !important;
            text-align: left !important;
            font-size: 1rem !important;
            padding: 0.3rem 0 !important;
            margin: 0 !important;
            width: 100%;
            cursor: pointer;
        }
        .sample-question-btn:hover {
            text-decoration: underline;
            background: none !important;
        }
        /* Clear Chat History button */
        .clear-chat-btn button {
            background-color: #10316b !important; /* Dark blue */
            color: #fff !important;
            border-radius: 8px !important;
            padding: 10px 18px !important;
            font-weight: bold !important;
            width: 100%;
            margin-bottom: 1rem;
            border: none !important;
            font-size: 1rem !important;
        }
        /* Chat message bubbles */
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

def save_chat_history(messages):
    with open("chat_history.txt", "a", encoding="utf-8") as f:
        for msg in messages:
            f.write(f"{msg['role']}: {msg['content']}\n")
        f.write("\n---\n")

def main():
    st.title("Mental Health Assistant")

    # Sidebar
    with st.sidebar:
        st.header("Options")
        store_data = st.checkbox("Store chat data", value=True)
        # Clear Chat History button with custom class
        clear_chat = st.button("Clear Chat History", key="clear_chat", help="Clear all chat history")
        st.markdown(
            '<style>.clear-chat-btn button{background-color:#10316b!important;color:#fff!important;border-radius:8px!important;padding:10px 18px!important;font-weight:bold!important;width:100%;margin-bottom:1rem;border:none!important;font-size:1rem!important;}</style>',
            unsafe_allow_html=True
        )
        # Hack: Add class to the button using HTML (Streamlit doesn't support button class directly)
        st.markdown(
            """
            <style>
            div[data-testid="stSidebar"] button[kind="secondary"] {
                background-color: #10316b !important;
                color: #fff !important;
                border-radius: 8px !important;
                padding: 10px 18px !important;
                font-weight: bold !important;
                width: 100%;
                margin-bottom: 1rem;
                border: none !important;
                font-size: 1rem !important;
            }
            </style>
            """, unsafe_allow_html=True
        )
        if clear_chat:
            st.session_state.messages = []
            st.success("Chat history cleared.")

        st.markdown("### Sample Questions")
        sample_questions = [
            "What are common signs of anxiety?",
            "How can I manage stress?",
            "What should I do if I feel depressed?",
            "Where can I find mental health resources?"
        ]
        # Render sample questions as clickable text (not buttons)
        for idx, q in enumerate(sample_questions):
            if f'sample_{idx}' not in st.session_state:
                st.session_state[f'sample_{idx}'] = False
            if st.button(q, key=f"sample_{idx}_hidden", help=f"Ask: {q}", args=(q,), kwargs={}, use_container_width=True):
                st.session_state['sample_prompt'] = q

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Chat input
    prompt = st.chat_input("Type your question about mental health...")

    # Check if a sample question was selected
    if 'sample_prompt' in st.session_state:
        prompt = st.session_state['sample_prompt']
        del st.session_state['sample_prompt']

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        with st.spinner("Thinking..."):
            try:
                HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
                HF_TOKEN = os.getenv("HF_TOKEN")

                vectorstore = get_vectorstore()
                if vectorstore:
                    llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
                    qa_chain = get_qa_chain(llm, vectorstore)

                    response = qa_chain.invoke({'query': prompt})
                    result = response["result"]

                    st.chat_message('assistant').markdown(result)
                    st.session_state.messages.append(
                        {'role': 'assistant', 'content': result}
                    )
                    if store_data:
                        save_chat_history([
                            {'role': 'user', 'content': prompt},
                            {'role': 'assistant', 'content': result}
                        ])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
