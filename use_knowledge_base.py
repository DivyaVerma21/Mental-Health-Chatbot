import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"


def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
        max_new_tokens=512,
        return_full_text=False  # Moved out of model_kwargs
    )


# Optimized Prompt Template
CUSTOM_PROMPT_TEMPLATE = """[INST]
Answer the question based only on the provided context.
If you don't know the answer, say "I don't know".
Keep answers concise and factual.

Context: {context}
Question: {input}

Answer: [/INST]"""

# Load FAISS Database
DB_FAISS_PATH = "vectorstore/db_faiss"
if not os.path.exists(DB_FAISS_PATH):
    raise FileNotFoundError(f"FAISS database not found at {DB_FAISS_PATH}")

# Initialize components
llm = load_llm()
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Load vector store
db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 3}
)

# Create chains
prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "input"]
)

document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)


# Query processing
def get_response(query):
    try:
        response = qa_chain.invoke({"input": query})
        return {
            "answer": response["answer"],
            "sources": [doc.metadata.get("source", "Unknown")
                        for doc in response["context"]]
        }
    except Exception as e:
        return {"error": str(e)}


# Main execution
if __name__ == "__main__":
    print("Mental Health Chatbot (Type 'quit' to exit)")
    while True:
        user_query = input("\nYour question: ").strip()
        if user_query.lower() in ('quit', 'exit'):
            break

        if not user_query:
            print("Please enter a valid question.")
            continue

        result = get_response(user_query)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nAnswer: {result['answer']}")
            if result["sources"]:
                print("\nSources:")
                for src in set(result["sources"]):  # Remove duplicates
                    print(f"- {src}")
