import streamlit as st
import pandas as pd
import os
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings

# --- Ollama Configuration ---
try:
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="llama3")
except Exception as e:
    st.error(f"Failed to connect to Ollama. Is the Ollama application running? Details: {e}")
    st.stop()


@st.cache_resource
def load_rag_chain():
    """
    Loads the knowledge base, creates embeddings, and sets up the RAG chain.
    """
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        csv_path = os.path.join(script_dir, 'knowledge_base.csv')

        # --- DIAGNOSTIC STEP: Test loading with pandas first ---
        st.info(f"Attempting to load knowledge base from: {csv_path}")
        try:
            pd.read_csv(csv_path, encoding='utf-8')
            st.success("Successfully pre-loaded and validated CSV file with pandas.")
        except Exception as pandas_error:
            st.error(f"Pandas failed to read the CSV file. Please check the file's content and format. Error: {pandas_error}")
            return None
        # --- END OF DIAGNOSTIC STEP ---

        # --- FIXED: Changed "issue_summary" to the correct column name "dasher_query" ---
        loader = CSVLoader(file_path=csv_path, source_column="dasher_query", encoding='utf-8')
        documents = loader.load()

        if not documents:
            st.error("The knowledge_base.csv file is empty or could not be read by the loader. Please ensure it has content.")
            return None

        vector_store = Chroma.from_documents(documents, embedding_model)
        retriever = vector_store.as_retriever()

        prompt = ChatPromptTemplate.from_template("""
        Answer the user's question based only on the context provided.
        If the context doesn't contain the answer, state that you don't have enough information.
        Context: {context}
        Question: {input}
        Answer:
        """)

        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        print("RAG chain ready.")
        return retrieval_chain

    except Exception as e:
        st.error(f"An unexpected error occurred while loading the RAG chain: {e}")
        return None

# --- Streamlit UI ---
st.title("Dasher Support Chatbot (Ollama Powered) 🤖")
st.caption("This chatbot uses a local Llama 3 model to answer questions based on a knowledge base.")

rag_chain = load_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about a Dasher issue..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": prompt})
                answer = response.get("answer", "Sorry, something went wrong.")
                st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("RAG chain is not available. Cannot process the request.")

