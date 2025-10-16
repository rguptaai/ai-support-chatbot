RAG-Based Dasher Support Chatbot

This project is a comprehensive, multi-stage chatbot system designed to provide support for delivery drivers ("Dashers"), inspired by DoorDash's AI support architecture.

The system is built using a Retrieval-Augmented Generation (RAG) pipeline and includes real-time safety checks (Guardrail) and an offline performance evaluation module (Judge).

System Architecture

The project is divided into three main components:

Core RAG System (rag_system.py):

Loads a knowledge base from knowledge_base.csv.

Uses Google's Gemini models for embeddings and text generation via LangChain.

Creates a searchable vector store using ChromaDB.

Provides a command-line interface for asking questions and receiving context-aware answers.

Logs all interactions to evaluation_log.csv for later analysis.

LLM Guardrail (guardrail.py):

A real-time safety and compliance filter.

Uses a Gemini model with a specific prompt to check if a generated response violates predefined company policies (e.g., giving financial advice, using unprofessional language).

Ensures that only safe and appropriate responses are sent to the user.

LLM Judge (judge.py):

An offline evaluation system that measures the quality of the chatbot's performance over time.

Uses the ragas library with Gemini as the judging LLM.

Reads the evaluation_log.csv file and calculates key RAG metrics like faithfulness, answer_relevancy, context_precision, and context_recall.

Outputs a detailed performance report to evaluation_results.csv.

How to Run This Project

1. Prerequisites:

Python 3.8+

Google Cloud Project with Vertex AI enabled.

2. Setup:

Clone the repository.

Install the required Python libraries:

pip install pandas langchain-google-vertexai langchain-community "langchain[chromadb]" ragas datasets


Authenticate with Google Cloud:

gcloud auth application-default login


Important: In rag_system.py and judge.py, replace "YOUR_GOOGLE_CLOUD_PROJECT_ID" with your actual Google Cloud Project ID.

3. Running the Chatbot:

To run the chatbot with the integrated guardrail, execute:

python rag_system.py


Interact with the chatbot in your terminal. When you're done, type exit to save the interaction log.

4. Evaluating Performance:

Once evaluation_log.csv has been created, run the judge:

python judge.py


Check the terminal for the performance summary and review evaluation_results.csv for a detailed breakdown.