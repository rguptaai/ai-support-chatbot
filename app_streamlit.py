import streamlit as st
import pandas as pd
import os
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, Docx2txtLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
import csv

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
    Scans a directory for knowledge base files, loads them, creates embeddings,
    and sets up the RAG chain.
    """
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        knowledge_base_dir = os.path.join(script_dir, 'knowledge_base_sources')
        
        all_documents = []
        
        st.info(f"Scanning for knowledge files in: {knowledge_base_dir}")

        for filename in os.listdir(knowledge_base_dir):
            file_path = os.path.join(knowledge_base_dir, filename)
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                all_documents.extend(loader.load())
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                all_documents.extend(loader.load())
            elif filename.endswith(".csv"):
                loader = CSVLoader(file_path=file_path, source_column="dasher_query", encoding='utf-8')
                all_documents.extend(loader.load())

        if not all_documents:
            st.error("No documents found in 'knowledge_base_sources' folder.")
            return None

        st.success(f"Successfully loaded {len(all_documents)} documents.")

        vector_store = Chroma.from_documents(all_documents, embedding_model)
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
        
        print("RAG chain ready with multi-source knowledge base.")
        return retrieval_chain

    except Exception as e:
        st.error(f"An unexpected error occurred while loading the RAG chain: {e}")
        return None

# --- Function to log interactions (unchanged) ---
def log_interaction(question, contexts, answer):
    log_file = 'evaluation_log.csv'
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['question', 'contexts', 'answer'])
        contexts_str = [doc.page_content for doc in contexts]
        writer.writerow([question, contexts_str, answer])

# --- Streamlit UI ---
st.title("Dasher Support Chatbot (with Citations) 🤖")
st.caption("This chatbot can learn from CSV, PDF, and Word documents and cites its sources.")

rag_chain = load_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": prompt})
                answer = response.get("answer", "Sorry, something went wrong.")
                
                # --- NEW: Extract and format source citations ---
                contexts = response.get("context", [])
                sources = set() # Use a set to store unique source filenames
                if contexts:
                    for doc in contexts:
                        source_path = doc.metadata.get('source', 'Unknown')
                        sources.add(os.path.basename(source_path)) # Get just the filename
                
                # Format the final answer with sources
                if sources:
                    answer_with_sources = f"{answer}\n\n---\n*Sources: {', '.join(sources)}*"
                else:
                    answer_with_sources = answer

                st.markdown(answer_with_sources)
                
                # Log the original interaction for evaluation
                log_interaction(prompt, contexts, answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer_with_sources})
    else:
        st.error("RAG chain is not available. Cannot process the request.")
