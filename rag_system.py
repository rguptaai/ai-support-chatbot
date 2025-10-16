import os
import pandas as pd
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Configuration ---
# Make sure you have authenticated with Google Cloud CLI:
# gcloud auth application-default login
#
# IMPORTANT: Replace "YOUR_GOOGLE_CLOUD_PROJECT_ID" with your actual Project ID.
os.environ["GCLOUD_PROJECT"] = "YOUR_GOOGLE_CLOUD_PROJECT_ID" 

# --- 2. Initialize Models ---
print("Initializing Gemini models...")
# Initialize the Gemini embedding model to turn text into numbers
embedding_model = VertexAIEmbeddings(model_name="textembedding-gecko@003")

# Initialize the Gemini generative model to create answers
llm = ChatVertexAI(model_name="gemini-1.0-pro")
print("Models initialized.")

# --- 3. Load and Process Knowledge Base ---
print("Loading knowledge base from CSV...")
loader = CSVLoader(file_path='./knowledge_base.csv', source_column='dasher_query')
documents = loader.load()

# Split the documents into smaller, manageable chunks for the database
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

print(f"Loaded and split {len(docs)} document chunks.")

# --- 4. Create Vector Store ---
# This is the "brain" of the RAG system. It creates a searchable database (ChromaDB)
# of your document chunks.
print("Creating vector store... This may take a moment.")
vector_store = Chroma.from_documents(docs, embedding_model)

# Create a retriever which is a tool to find the most relevant document chunks
retriever = vector_store.as_retriever(search_kwargs={"k": 2}) # Retrieve top 2 most relevant chunks
print("Vector store created successfully.")

# --- 5. Define the RAG Chain using LangChain Expression Language (LCEL) ---
# This is the prompt template that tells the LLM how to behave.
template = """
You are a helpful assistant for DoorDash delivery drivers (Dashers).
Answer the user's question based only on the following context.
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# This is the complete RAG chain. It defines the flow of data:
# 1. The user's question is passed to the retriever.
# 2. The retriever finds relevant documents and formats them.
# 3. The documents and the original question are fed into the prompt.
# 4. The prompt is sent to the LLM (Gemini).
# 5. The LLM's response is parsed into a simple string.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 6. Interactive Chatbot Loop ---
if __name__ == "__main__":
    print("\n--- Dasher Support Chatbot ---")
    print("Ask a question, or type 'exit' to quit.")

    # We will log interactions to this list for later evaluation
    evaluation_log = []

    while True:
        # Use input() for Python 3
        query = input("You: ")
        if query.lower() == 'exit':
            break

        # Invoke the RAG chain to get a response
        response = rag_chain.invoke(query)
        print("Bot:", response)
        
        # --- Logging for the LLM Judge ---
        # Retrieve the documents again to log them
        retrieved_docs = retriever.get_relevant_documents(query)
        
        # For evaluation, we need a "ground truth" answer.
        # In a real system, this is the ideal answer.
        # Here, we'll just pull the original resolution from the CSV metadata.
        ground_truth = ""
        if retrieved_docs:
            ground_truth = retrieved_docs[0].metadata.get('resolution', "No ground truth available.")

        evaluation_log.append({
            "question": query,
            "answer": response,
            "contexts": [doc.page_content for doc in retrieved_docs],
            "ground_truth": ground_truth
        })

    # Save the log to a CSV file when the chat ends
    print("\nSaving conversation log for evaluation...")
    log_df = pd.DataFrame(evaluation_log)
    log_df.to_csv("evaluation_log.csv", index=False)
    print("Evaluation log saved to evaluation_log.csv")


