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
    page_title="MindBridge Assistant",
    layout="centered",
    page_icon="🌍"
)

# Custom CSS styling with dual-theme colors
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
        st.error(f"Knowledge base loading error: {str(e)}")
        return None


def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.3,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
        max_new_tokens=512,
        return_full_text=False
    )


def get_qa_chain(llm, vectorstore):
    CUSTOM_PROMPT_TEMPLATE = """[INST]
    You are a compassionate mental health assistant providing culturally-sensitive support for India and Norway.
    For India responses, consider: family dynamics, stigma issues, and available resources.
    For Norway responses, emphasize: social welfare systems, English-speaking services, and quick access to care.
    Always suggest professional help when needed and provide local emergency contacts when relevant.

    Context: {context}
    Question: {question}

    Provide a thoughtful, location-appropriate response: [/INST]"""

    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 4}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )


def display_resources():
    tab1, tab2 = st.tabs(["🇮🇳 India Resources", "🇳🇴 Norway Resources"])

    with tab1:
        st.markdown("""
        ### Emergency Contacts:
        - **Vandrevala Foundation**: +91 9999 666 555 (24/7)
        - **iCall**: 022-25521111 (Mon-Sat, 8AM-10PM)
        - **Fortis Stress Helpline**: +91-8376804102

        ### Counseling Services:
        - **The Mind Clinic** (Delhi/Mumbai/Bangalore)
        - **Parivarthan** (Bangalore)
        - **1to1help.net** (Online counseling)

        ### Digital Tools:
        - **Wysa** AI chatbot
        - **YourDOST** online platform
        """)

    with tab2:
        st.markdown("""
        ### Emergency Contacts:
        - **Mental Helse**: 116 123
        - **Legevakt** (Emergency): 116 117
        - **SOS International**: +47 67 59 71 71 (English)

        ### English-Speaking Therapists:
        - **Psykolog Sentrum** (Oslo)
        - **The Oak Clinic** (Oslo)
        - **Norwegian Institute of Cognitive Therapy**

        ### Public Services:
        - **DPS** (District Psychiatric Centers)
        - **Fastlege** (GP referral system)
        - **Helsenorge.no** official health portal
        """)


def main():
    # App header
    st.title("MindBridge Assistant")
    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h3 style='color: #5E35B1;'>Bridging mental health support for India and Norway</h3>
            <p>Select your location for tailored resources</p>
        </div>
    """, unsafe_allow_html=True)

    display_resources()

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = [{
            'role': 'assistant',
            'content': "Hello! I provide mental health support for both India and Norway. Where are you located today?"
        }]

    # Display chat messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Chat input
    if prompt := st.chat_input("Ask about mental health resources..."):
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

                # Show crisis resources if needed
                if any(word in prompt.lower() for word in ['suicide', 'emergency', 'helpline']):
                    with st.expander("🚨 Immediate Crisis Support"):
                        st.markdown("""
                        **India**: Call Vandrevala Foundation at +91 9999 666 555  
                        **Norway**: Call 113 for life-threatening emergencies or 116 123 for mental health support
                        """)
        except Exception as e:
            st.error(f"Sorry, I encountered an error. Please try rephrasing your question.")


if __name__ == "__main__":
    main()
